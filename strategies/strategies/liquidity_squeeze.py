"""Liquidity Squeeze strategy.

When the bid-ask spread suddenly widens far beyond normal, it signals
a temporary liquidity crisis. Prices deviate from fair value during
these events and typically snap back once liquidity returns.

Signal:
  - Current spread > 3x the configured max_spread_pct baseline
  - Recent price movement away from mid (confirming the squeeze)
  - Volume above minimum (squeeze in active markets, not dead ones)

Direction: fade the price move (bet on reversion).
Not traded in trending regimes (avoid fighting momentum).
"""

from __future__ import annotations

import structlog

from config import settings
from core.state import PortfolioState
from strategies.base import BaseStrategy, MarketContext, Signal

logger = structlog.get_logger(__name__)

SPREAD_MULTIPLIER = 3.0
MIN_SPREAD_ABS = 0.02
RESOLUTION_GUARD = 0.05
MIN_PRICE_MOVE = 0.01


class LiquiditySqueezeStrategy(BaseStrategy):
    """Mean reversion on temporary liquidity squeeze events."""

    @property
    def name(self) -> str:
        return "liquidity_squeeze"

    async def generate_signal(self, context: MarketContext) -> Signal | None:
        market = context.market
        ob = context.orderbook
        history = context.probability_history

        if ob is None or len(history) < 6:
            return None

        prob = market.probability
        if prob < RESOLUTION_GUARD or prob > (1 - RESOLUTION_GUARD):
            return None

        if market.volume_24h < settings.min_daily_volume:
            return None

        # Get orderbook spread
        best_bid = getattr(ob, "best_bid", 0.0) or 0.0
        best_ask = getattr(ob, "best_ask", 0.0) or 0.0

        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return None

        current_spread = best_ask - best_bid

        if current_spread < MIN_SPREAD_ABS:
            return None

        # Normal spread baseline from settings
        normal_spread = (settings.max_spread_pct / 100) / 3  # typical is 1/3 of max
        if normal_spread <= 0:
            normal_spread = 0.015

        if current_spread < normal_spread * SPREAD_MULTIPLIER:
            logger.debug(
                "liquidity_squeeze_rejected",
                market_id=market.market_id,
                reason="spread_normal",
                current=round(current_spread, 4),
                threshold=round(normal_spread * SPREAD_MULTIPLIER, 4),
            )
            return None

        # Need a directional price move to fade
        recent_change = history[-1] - history[-6]
        if abs(recent_change) < MIN_PRICE_MOVE:
            return None

        # Fade the move
        direction = "SELL" if recent_change > 0 else "BUY"
        spread_ratio = current_spread / normal_spread
        strength = min((spread_ratio - SPREAD_MULTIPLIER) / SPREAD_MULTIPLIER, 1.0)
        strength = max(strength, 0.1)

        signal = Signal(
            market_id=market.market_id,
            condition_id=market.condition_id,
            token_id=market.token_ids[0] if market.token_ids else "",
            direction=direction,
            strength=strength,
            strategy=self.name,
            category=market.category,
            metadata={
                "current_spread": round(current_spread, 4),
                "normal_spread": round(normal_spread, 4),
                "spread_ratio": round(spread_ratio, 2),
                "recent_change": round(recent_change, 4),
            },
        )

        logger.info(
            "liquidity_squeeze_signal",
            market_id=market.market_id,
            direction=direction,
            spread_ratio=round(spread_ratio, 2),
            recent_change=round(recent_change, 4),
        )
        return signal

    def calculate_size(self, signal: Signal, portfolio: PortfolioState) -> float:
        # Slightly smaller size — squeeze trades are mean-reversion bets
        return portfolio.capital * (settings.default_position_pct / 100) * 0.7
