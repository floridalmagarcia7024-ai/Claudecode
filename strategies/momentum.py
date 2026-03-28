"""Momentum strategy (Module 11).

Only active in TRENDING_UP or TRENDING_DOWN regimes.

Entry signal:
  - 1h price change >= 0.025
  - 1h volume > 1.5x 7-day average volume
  - Optional AI confirmation boosts sizing

Anti-resolution filter: ignores signals if market resolves
in <48h and probability is converging to 0 or 1.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

import structlog

from config import settings
from core.regime import MarketRegime
from core.state import PortfolioState
from strategies.base import BaseStrategy, MarketContext, Signal

logger = structlog.get_logger(__name__)

MOMENTUM_CHANGE_THRESHOLD = 0.025
VOLUME_MULTIPLIER = 1.5
MOMENTUM_EXIT_THRESHOLD = 0.01
CONVERGENCE_THRESHOLD = 0.10  # prob near 0 or 1


class MomentumStrategy(BaseStrategy):
    """Momentum strategy for trending markets."""

    def __init__(self, regime_detector: object | None = None) -> None:
        self._regime_detector = regime_detector

    @property
    def name(self) -> str:
        return "momentum"

    async def generate_signal(self, context: MarketContext) -> Signal | None:
        """Generate momentum signal if conditions are met.

        Filters:
            1. Regime must be TRENDING_UP or TRENDING_DOWN
            2. 1h change >= 0.025
            3. Volume > 1.5x average
            4. Anti-resolution filter
        """
        market = context.market
        history = context.probability_history

        # Need at least 12 data points (1h at 5-min intervals)
        if len(history) < 12:
            return None

        # Check regime
        regime = self._get_regime(market.market_id)
        if regime not in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return None

        # Calculate 1h change
        current_prob = market.probability
        prob_1h_ago = history[-12] if len(history) >= 12 else history[0]
        change_1h = current_prob - prob_1h_ago

        if abs(change_1h) < MOMENTUM_CHANGE_THRESHOLD:
            logger.debug(
                "momentum_rejected",
                market_id=market.market_id,
                reason="change_below_threshold",
                change_1h=round(change_1h, 4),
            )
            return None

        # Volume check: compare current volume to 7-day average
        volume_avg_7d = context.metadata.get("volume_avg_7d", 0.0) if hasattr(context, 'metadata') else 0.0
        if volume_avg_7d <= 0:
            # If no 7d average available, use a basic heuristic
            volume_avg_7d = settings.min_daily_volume

        if market.volume_24h < volume_avg_7d * VOLUME_MULTIPLIER:
            logger.debug(
                "momentum_rejected",
                market_id=market.market_id,
                reason="volume_below_threshold",
                volume_24h=market.volume_24h,
                required=round(volume_avg_7d * VOLUME_MULTIPLIER, 0),
            )
            return None

        # Anti-resolution filter
        if self._is_converging_pre_resolution(market, current_prob):
            logger.info(
                "momentum_rejected",
                market_id=market.market_id,
                reason="anti_resolution_filter",
                probability=round(current_prob, 4),
                end_date=market.end_date,
            )
            return None

        direction = "BUY" if change_1h > 0 else "SELL"
        strength = min(abs(change_1h) / (MOMENTUM_CHANGE_THRESHOLD * 3), 1.0)

        signal = Signal(
            market_id=market.market_id,
            condition_id=market.condition_id,
            token_id=market.token_ids[0] if market.token_ids else "",
            direction=direction,
            strength=strength,
            strategy=self.name,
            category=market.category,
            metadata={
                "change_1h": round(change_1h, 4),
                "volume_24h": market.volume_24h,
                "regime": regime.value,
            },
        )

        logger.info(
            "momentum_signal_generated",
            market_id=market.market_id,
            direction=direction,
            change_1h=round(change_1h, 4),
            strength=round(strength, 3),
            regime=regime.value,
        )
        return signal

    def calculate_size(self, signal: Signal, portfolio: PortfolioState) -> float:
        """Basic fallback sizing — actual sizing done by RiskManager."""
        return portfolio.capital * (settings.default_position_pct / 100)

    @staticmethod
    def check_exit_conditions(change_1h: float) -> bool:
        """Exit when momentum fades (change < 0.01)."""
        return abs(change_1h) < MOMENTUM_EXIT_THRESHOLD

    def _get_regime(self, market_id: str) -> MarketRegime:
        """Get regime from detector or default to RANGING."""
        if self._regime_detector is not None and hasattr(self._regime_detector, "get_regime"):
            result = self._regime_detector.get_regime(market_id)
            if isinstance(result, MarketRegime):
                return result
        return MarketRegime.RANGING

    @staticmethod
    def _is_converging_pre_resolution(market: object, current_prob: float) -> bool:
        """Check if market is converging to 0/1 near resolution.

        Filters out false momentum signals that are actually
        pre-resolution convergence.
        """
        end_date = getattr(market, "end_date", "")
        if not end_date:
            return False

        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            hours_until = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        except (ValueError, TypeError):
            return False

        if hours_until > 48:
            return False

        # Near resolution AND probability converging to 0 or 1
        return current_prob < CONVERGENCE_THRESHOLD or current_prob > (1 - CONVERGENCE_THRESHOLD)
