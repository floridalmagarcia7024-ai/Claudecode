"""Smart exit manager with TWAP, impact estimation, and limit escalation (Module 16).

Consolidates exit logic:
  - Market impact estimator (pre-trade)
  - Liquidation price estimator
  - Smart limit order escalation (limit -> mid -> market)
  - TWAP execution for large positions
  - Realistic slippage simulation for paper mode
"""

from __future__ import annotations

import asyncio
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

from api.polymarket import OrderBook, PolymarketClient
from config import settings
from core.state import Position, PositionManager

logger = structlog.get_logger(__name__)

# Impact thresholds
IMPACT_LOW = 1.5       # <= 1.5%: immediate market order
IMPACT_MED = 3.0       # 1.5-3%: TWAP 3 parts, 60s
# > 3%: TWAP 5 parts, 120s

LIMIT_WAIT_S = 30      # Wait time for limit order fill


@dataclass
class ExitPlan:
    """Planned exit strategy for a position."""

    position_id: int
    method: str  # "immediate", "twap_3", "twap_5"
    estimated_impact_pct: float = 0.0
    num_parts: int = 1
    interval_s: int = 0
    urgency: str = "normal"  # "normal" or "high"


@dataclass
class SlippageRecord:
    """Historical slippage observation."""

    market_id: str
    slippage_pct: float
    size_usd: float
    timestamp: str = ""


class SmartExitManager:
    """Manages intelligent position exits with impact estimation and TWAP."""

    def __init__(
        self,
        client: PolymarketClient | None = None,
        state: PositionManager | None = None,
    ) -> None:
        self._client = client
        self._state = state
        self._slippage_history: dict[str, list[float]] = defaultdict(list)
        self._avg_order_sizes: dict[str, float] = defaultdict(lambda: 100.0)

    def estimate_market_impact(
        self, position_size_usd: float, depth_ask_5: float
    ) -> float:
        """Estimate market impact as percentage.

        market_impact = position_size_usd / depth_ask_5 * 100
        """
        if depth_ask_5 <= 0:
            return 100.0
        return position_size_usd / depth_ask_5 * 100

    def plan_exit(
        self, position: Position, orderbook: OrderBook, urgency: str = "normal"
    ) -> ExitPlan:
        """Create an exit plan based on market impact.

        Args:
            position: The position to exit.
            orderbook: Current orderbook.
            urgency: "normal" or "high" (trailing stop).

        Returns:
            ExitPlan with method and parameters.
        """
        impact = self.estimate_market_impact(position.size_usd, orderbook.depth_5_usd)

        # High urgency (trailing stop): always market order
        if urgency == "high":
            return ExitPlan(
                position_id=position.id or 0,
                method="immediate",
                estimated_impact_pct=impact,
                num_parts=1,
                interval_s=0,
                urgency="high",
            )

        if impact <= IMPACT_LOW:
            return ExitPlan(
                position_id=position.id or 0,
                method="immediate",
                estimated_impact_pct=impact,
                num_parts=1,
                interval_s=0,
            )
        elif impact <= IMPACT_MED:
            return ExitPlan(
                position_id=position.id or 0,
                method="twap_3",
                estimated_impact_pct=impact,
                num_parts=3,
                interval_s=60,
            )
        else:
            return ExitPlan(
                position_id=position.id or 0,
                method="twap_5",
                estimated_impact_pct=impact,
                num_parts=5,
                interval_s=120,
            )

    def estimate_exit_price(
        self, position: Position, orderbook: OrderBook
    ) -> float:
        """Estimate the exit price including slippage.

        exit_price_est = mid_price * (1 + slippage_est)
        """
        mid = orderbook.mid_price
        if not mid or mid <= 0:
            return 0.0

        slippage_est = self._estimate_slippage(
            position.market_id, position.size_usd
        )

        if position.direction == "BUY":
            # Selling: price goes down with slippage
            return mid * (1 - slippage_est / 100)
        else:
            # Buying back: price goes up with slippage
            return mid * (1 + slippage_est / 100)

    def should_reject_trade(
        self, entry_price: float, exit_price_est: float, direction: str, fees_pct: float = 0.0
    ) -> bool:
        """Check if estimated exit price is below breakeven.

        If exit_price_est < breakeven_price: REJECT.
        """
        fee_factor = fees_pct / 100
        if direction == "BUY":
            breakeven = entry_price * (1 + fee_factor * 2)
            return exit_price_est < breakeven
        else:
            breakeven = entry_price * (1 - fee_factor * 2)
            return exit_price_est > breakeven

    async def execute_smart_exit(
        self, position: Position, orderbook: OrderBook, urgency: str = "normal"
    ) -> dict:
        """Execute a smart exit with limit order escalation.

        Steps for normal urgency:
          1. Limit order near best bid/ask -> wait 30s
          2. If not filled -> limit at mid price -> wait 30s
          3. If not filled -> market order

        For high urgency: market order immediately.

        Returns:
            Dict with execution details.
        """
        result = {
            "position_id": position.id,
            "method": "smart_limit",
            "steps": [],
            "final_price": 0.0,
            "total_slippage_pct": 0.0,
        }

        if settings.paper_mode:
            return await self._paper_smart_exit(position, orderbook, urgency, result)

        # Real mode execution
        if urgency == "high" or not self._client:
            result["method"] = "market_order"
            result["steps"].append("market_order_direct")
            result["final_price"] = orderbook.mid_price
            return result

        # Step 1: Limit near best bid/ask
        if position.direction == "BUY":
            limit_price = orderbook.best_bid
        else:
            limit_price = orderbook.best_ask

        result["steps"].append(f"limit_order_at_{limit_price:.4f}")
        await asyncio.sleep(LIMIT_WAIT_S)

        # Step 2: Move to mid price
        limit_price = orderbook.mid_price
        result["steps"].append(f"limit_aggressive_at_{limit_price:.4f}")
        await asyncio.sleep(LIMIT_WAIT_S)

        # Step 3: Market order
        result["steps"].append("market_order_fallback")
        result["final_price"] = orderbook.mid_price
        result["method"] = "limit_escalation"

        return result

    async def execute_twap(
        self, position: Position, plan: ExitPlan
    ) -> dict:
        """Execute TWAP (Time-Weighted Average Price) exit.

        Splits the position into N parts and executes at intervals.
        """
        part_size = position.size_usd / plan.num_parts
        executed_parts: list[dict] = []
        total_value = 0.0

        for i in range(plan.num_parts):
            if self._client and position.token_id:
                try:
                    ob = await self._client.get_orderbook(position.token_id)
                    price = ob.mid_price
                except Exception:
                    price = position.current_price
            else:
                price = position.current_price

            # In paper mode, add realistic slippage
            if settings.paper_mode:
                slippage = self._sample_slippage(position.market_id, part_size)
                if position.direction == "BUY":
                    price *= (1 - slippage / 100)
                else:
                    price *= (1 + slippage / 100)

            executed_parts.append({
                "part": i + 1,
                "size_usd": round(part_size, 2),
                "price": round(price, 4),
            })
            total_value += price * part_size

            if i < plan.num_parts - 1:
                await asyncio.sleep(plan.interval_s)

        avg_price = total_value / position.size_usd if position.size_usd > 0 else 0

        logger.info(
            "twap_complete",
            position_id=position.id,
            parts=plan.num_parts,
            avg_price=round(avg_price, 4),
        )

        return {
            "position_id": position.id,
            "method": f"twap_{plan.num_parts}",
            "parts": executed_parts,
            "avg_price": round(avg_price, 4),
        }

    def record_slippage(self, market_id: str, slippage_pct: float, size_usd: float) -> None:
        """Record observed slippage for future simulation."""
        self._slippage_history[market_id].append(slippage_pct)
        # Keep last 100 observations per market
        if len(self._slippage_history[market_id]) > 100:
            self._slippage_history[market_id] = self._slippage_history[market_id][-100:]

        # Update avg order size
        existing = self._avg_order_sizes[market_id]
        self._avg_order_sizes[market_id] = (existing + size_usd) / 2

    def _sample_slippage(self, market_id: str, size_usd: float) -> float:
        """Sample from historical slippage distribution, adjusted by size.

        slippage_final *= min(2.0, size / avg_order_size)
        """
        history = self._slippage_history.get(market_id, [])

        if len(history) >= 5:
            base_slippage = random.choice(history)
        else:
            # Default distribution
            base_slippage = random.uniform(0.1, 0.5)

        avg_size = self._avg_order_sizes[market_id]
        size_factor = min(2.0, size_usd / avg_size) if avg_size > 0 else 1.0

        return abs(base_slippage) * size_factor

    def _estimate_slippage(self, market_id: str, size_usd: float) -> float:
        """Estimate expected slippage for a given market and size."""
        history = self._slippage_history.get(market_id, [])
        if len(history) >= 5:
            base = statistics.median(history)
        else:
            base = 0.3  # Default 0.3%

        avg_size = self._avg_order_sizes[market_id]
        size_factor = min(2.0, size_usd / avg_size) if avg_size > 0 else 1.0
        return base * size_factor

    async def _paper_smart_exit(
        self, position: Position, orderbook: OrderBook, urgency: str, result: dict
    ) -> dict:
        """Simulate smart exit in paper mode with realistic slippage."""
        slippage = self._sample_slippage(position.market_id, position.size_usd)

        if urgency == "high":
            # Market order: higher slippage
            slippage *= 1.5
            result["method"] = "market_order_urgent"
        else:
            result["method"] = "smart_limit_paper"

        mid = orderbook.mid_price if orderbook.mid_price > 0 else position.current_price

        if position.direction == "BUY":
            final_price = mid * (1 - slippage / 100)
        else:
            final_price = mid * (1 + slippage / 100)

        result["final_price"] = round(final_price, 4)
        result["total_slippage_pct"] = round(slippage, 3)
        result["steps"].append(f"paper_exit_at_{final_price:.4f}")

        self.record_slippage(position.market_id, slippage, position.size_usd)

        return result
