"""Paper trading engine with simulated execution.

Simulates order fills with variable slippage, latency, and fees.
All trades are recorded in SQLite via PositionManager.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import structlog

from api.polymarket import OrderBook
from config import settings
from core.state import Position, PositionManager, PositionStatus

logger = structlog.get_logger(__name__)


@dataclass
class PaperFill:
    """Result of a simulated paper trade execution."""

    token_id: str
    direction: str
    requested_price: float
    fill_price: float
    size_usd: float
    fees_usd: float
    slippage_pct: float
    latency_ms: int
    position_id: int | None = None


class PaperTradingEngine:
    """Simulates order execution with realistic friction.

    - Variable slippage based on spread + random component
    - Simulated latency: 50-200ms
    - Fees: PAPER_FEE_PCT on entry + exit
    """

    def __init__(self, position_manager: PositionManager) -> None:
        self._pm = position_manager

    async def execute_order(
        self,
        token_id: str,
        direction: str,
        size_usd: float,
        orderbook: OrderBook | None,
        market_id: str = "",
        condition_id: str = "",
        strategy: str = "",
        category: str = "other",
    ) -> PaperFill:
        """Simulate order execution with slippage and latency.

        Args:
            token_id: Token to trade.
            direction: "BUY" or "SELL".
            size_usd: Position size in USD.
            orderbook: Current order book (for realistic fill price).
            market_id: Market identifier.
            condition_id: Condition identifier.
            strategy: Strategy name.
            category: Market category.

        Returns:
            PaperFill with execution details.
        """
        # Simulate latency (50-200ms)
        latency_ms = random.randint(50, 200)
        await asyncio.sleep(latency_ms / 1000)

        # Calculate fill price with slippage
        if orderbook and orderbook.mid_price > 0:
            mid = orderbook.mid_price
            spread_component = orderbook.spread_pct / 200  # Half spread as base slippage
            random_component = random.uniform(0, 0.005)  # 0-0.5% random noise
            slippage_pct = spread_component + random_component

            if direction == "BUY":
                fill_price = mid * (1 + slippage_pct / 100)
            else:
                fill_price = mid * (1 - slippage_pct / 100)
        else:
            fill_price = 0.50  # Default mid-market
            slippage_pct = 0.0

        # Calculate fees (entry fee)
        fees_usd = size_usd * (settings.paper_fee_pct / 100)

        # Record position
        position = Position(
            market_id=market_id,
            condition_id=condition_id,
            token_id=token_id,
            direction=direction,
            entry_price=fill_price,
            current_price=fill_price,
            size_usd=size_usd,
            status=PositionStatus.OPEN,
            strategy=strategy,
            category=category,
        )
        position_id = await self._pm.open_position(position)

        fill = PaperFill(
            token_id=token_id,
            direction=direction,
            requested_price=orderbook.mid_price if orderbook else fill_price,
            fill_price=fill_price,
            size_usd=size_usd,
            fees_usd=fees_usd,
            slippage_pct=slippage_pct,
            latency_ms=latency_ms,
            position_id=position_id,
        )

        logger.info(
            "paper_order_filled",
            market_id=market_id,
            direction=direction,
            fill_price=round(fill_price, 4),
            size_usd=round(size_usd, 2),
            fees_usd=round(fees_usd, 2),
            slippage_pct=round(slippage_pct, 3),
            latency_ms=latency_ms,
        )
        return fill

    async def close_position(
        self,
        position_id: int,
        current_price: float,
        exit_reason: str,
        orderbook: OrderBook | None = None,
    ) -> PaperFill | None:
        """Simulate closing a paper position.

        Args:
            position_id: ID of the position to close.
            current_price: Current market price.
            exit_reason: Reason for closing.
            orderbook: Order book for realistic fill simulation.

        Returns:
            PaperFill with close details, or None if position not found.
        """
        position = await self._pm.get_position_by_id(position_id)
        if position is None:
            logger.warning("paper_close_not_found", position_id=position_id)
            return None

        # Simulate close slippage
        if orderbook and orderbook.mid_price > 0:
            mid = orderbook.mid_price
            slippage_pct = orderbook.spread_pct / 200 + random.uniform(0, 0.005)
            # Closing: opposite direction slippage
            if position.direction == "BUY":
                exit_price = mid * (1 - slippage_pct / 100)
            else:
                exit_price = mid * (1 + slippage_pct / 100)
        else:
            exit_price = current_price
            slippage_pct = 0.0

        # Exit fees
        fees_usd = position.size_usd * (settings.paper_fee_pct / 100)

        # Record close
        await self._pm.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            fees_usd=fees_usd,
        )

        latency_ms = random.randint(50, 200)
        await asyncio.sleep(latency_ms / 1000)

        fill = PaperFill(
            token_id=position.token_id,
            direction="SELL" if position.direction == "BUY" else "BUY",
            requested_price=current_price,
            fill_price=exit_price,
            size_usd=position.size_usd,
            fees_usd=fees_usd,
            slippage_pct=slippage_pct,
            latency_ms=latency_ms,
            position_id=position_id,
        )

        logger.info(
            "paper_position_closed",
            position_id=position_id,
            exit_price=round(exit_price, 4),
            exit_reason=exit_reason,
            fees_usd=round(fees_usd, 2),
        )
        return fill

    async def check_real_mode_readiness(self) -> tuple[bool, list[str]]:
        """Validate if the bot is ready to transition to real trading.

        Requirements:
            1. At least 7 days of paper trading
            2. At least 10 paper trades completed
            3. Paper mode currently active

        Returns:
            Tuple of (ready, list_of_issues).
        """
        issues: list[str] = []

        if not settings.paper_mode:
            issues.append("Already in real mode")
            return False, issues

        # Check trade count
        total_trades = await self._pm.get_total_trade_count()
        if total_trades < 10:
            issues.append(f"Need 10+ paper trades, have {total_trades}")

        # Check trading duration
        trades = await self._pm.get_trade_history(limit=1000)
        if trades:
            oldest = min(t.timestamp for t in trades)
            oldest_dt = datetime.fromisoformat(oldest)
            days_trading = (datetime.now(timezone.utc) - oldest_dt).days
            if days_trading < 7:
                issues.append(f"Need 7+ days of paper trading, have {days_trading}")
        else:
            issues.append("No paper trades recorded yet")

        # Check API credentials
        if not settings.polymarket_api_key:
            issues.append("POLYMARKET_API_KEY not configured")
        if not settings.polymarket_secret:
            issues.append("POLYMARKET_SECRET not configured")
        if not settings.polymarket_private_key:
            issues.append("POLYMARKET_PRIVATE_KEY not configured")

        ready = len(issues) == 0
        if ready:
            logger.info("real_mode_ready", msg="All readiness checks passed")
        else:
            logger.info("real_mode_not_ready", issues=issues)

        return ready, issues
