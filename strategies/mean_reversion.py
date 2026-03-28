"""Mean reversion strategy using z-score on prediction market probabilities.

Signal: z_score = (current_prob - mean_7d) / std_7d
  BUY  if z_score < -ZSCORE_THRESHOLD
  SELL if z_score >  ZSCORE_THRESHOLD

LIMITATION: Probabilities in prediction markets are bounded [0,1] with
discontinuous jumps. The z-score is a heuristic, not an exact statistical measure.
"""

from __future__ import annotations

import statistics

import structlog

from config import settings
from core.state import PortfolioState
from intelligence.ai_analyzer import AIAnalyzer
from strategies.base import BaseStrategy, MarketContext, Signal

logger = structlog.get_logger(__name__)

# Minimum data points needed for a meaningful z-score
MIN_DATA_POINTS = 20  # ~7 days at 5-min intervals = 2016, but allow less for testing


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion on prediction market probabilities."""

    def __init__(self, ai_analyzer: AIAnalyzer | None = None) -> None:
        self._ai = ai_analyzer

    @property
    def name(self) -> str:
        return "mean_reversion"

    async def generate_signal(self, context: MarketContext) -> Signal | None:
        """Generate a signal if z-score exceeds threshold.

        Filters applied in order (stops at first rejection):
            1. Sufficient historical data
            2. Volume > MIN_DAILY_VOLUME
            3. Liquidity (spread, depth, slippage) — checked by engine
            4. Sentiment doesn't contradict direction (optional)
        """
        market = context.market
        history = context.probability_history

        # ── Filter 1: Sufficient data ──────────────────────────
        if len(history) < MIN_DATA_POINTS:
            logger.debug(
                "signal_rejected",
                market_id=market.market_id,
                reason="insufficient_data",
                data_points=len(history),
                required=MIN_DATA_POINTS,
            )
            return None

        # ── Compute z-score ────────────────────────────────────
        mean_7d = statistics.mean(history)
        std_7d = statistics.stdev(history)

        if std_7d < 0.001:
            logger.debug(
                "signal_rejected",
                market_id=market.market_id,
                reason="zero_variance",
                std=std_7d,
            )
            return None

        current_prob = market.probability
        z_score = (current_prob - mean_7d) / std_7d

        # ── Check threshold ────────────────────────────────────
        if abs(z_score) < settings.zscore_threshold:
            logger.debug(
                "signal_rejected",
                market_id=market.market_id,
                reason="zscore_below_threshold",
                z_score=round(z_score, 3),
                threshold=settings.zscore_threshold,
            )
            return None

        direction = "BUY" if z_score < -settings.zscore_threshold else "SELL"

        # ── Filter 2: Volume ───────────────────────────────────
        if market.volume_24h < settings.min_daily_volume:
            logger.info(
                "signal_rejected",
                market_id=market.market_id,
                reason="low_volume",
                volume_24h=market.volume_24h,
                min_required=settings.min_daily_volume,
                z_score=round(z_score, 3),
                direction=direction,
            )
            return None

        # ── Filter 3: Sentiment (optional) ─────────────────────
        sentiment_score = None
        ai_confidence = None

        if self._ai is not None:
            vader_result = self._ai.vader_analysis(market.question)
            sentiment_score = vader_result.confidence

            # Check if sentiment contradicts direction
            sentiment_direction = vader_result.direction
            contradicts = (
                (direction == "BUY" and sentiment_direction == "DOWN")
                or (direction == "SELL" and sentiment_direction == "UP")
            )

            if contradicts and vader_result.should_trade:
                logger.info(
                    "signal_rejected",
                    market_id=market.market_id,
                    reason="sentiment_contradicts",
                    direction=direction,
                    sentiment_direction=sentiment_direction,
                    sentiment_score=round(sentiment_score, 3),
                    z_score=round(z_score, 3),
                )
                return None

            # If VADER shows significant shift, get Groq analysis
            if abs(self._ai.analyze_sentiment(market.question)) > settings.sentiment_shift:
                groq_result = await self._ai.analyze_headline(
                    market.question, market.question, current_prob
                )
                ai_confidence = groq_result.confidence
                if not groq_result.should_trade:
                    logger.info(
                        "signal_rejected",
                        market_id=market.market_id,
                        reason="ai_rejects_trade",
                        ai_confidence=ai_confidence,
                        ai_direction=groq_result.direction,
                        z_score=round(z_score, 3),
                    )
                    return None

        # ── Signal confirmed ───────────────────────────────────
        strength = min(abs(z_score) / (settings.zscore_threshold * 2), 1.0)

        signal = Signal(
            market_id=market.market_id,
            condition_id=market.condition_id,
            token_id=market.token_ids[0] if market.token_ids else "",
            direction=direction,
            strength=strength,
            strategy=self.name,
            category=market.category,
            z_score=z_score,
            sentiment_score=sentiment_score,
            ai_confidence=ai_confidence,
            metadata={
                "mean_7d": round(mean_7d, 4),
                "std_7d": round(std_7d, 4),
                "current_prob": round(current_prob, 4),
                "volume_24h": market.volume_24h,
            },
        )

        logger.info(
            "signal_generated",
            market_id=market.market_id,
            direction=direction,
            z_score=round(z_score, 3),
            strength=round(strength, 3),
            volume_24h=market.volume_24h,
        )
        return signal

    def calculate_size(self, signal: Signal, portfolio: PortfolioState) -> float:
        """Delegate to risk manager — returns DEFAULT_POSITION_PCT * capital.

        The actual sizing with Kelly criterion and anti-martingale
        is handled by RiskManager.calculate_position_size().
        This provides a basic fallback.
        """
        return portfolio.capital * (settings.default_position_pct / 100)

    @staticmethod
    def check_exit_conditions(z_score: float) -> bool:
        """Check if position should be exited based on z-score mean reversion.

        Exit when z-score returns to [-0.5, 0.5] range.

        Args:
            z_score: Current z-score of the position's market.

        Returns:
            True if position should be exited.
        """
        return -0.5 <= z_score <= 0.5

    @staticmethod
    def compute_z_score(current: float, history: list[float]) -> float | None:
        """Compute z-score from current value and historical series.

        Args:
            current: Current probability.
            history: Historical probability series.

        Returns:
            Z-score or None if insufficient data.
        """
        if len(history) < MIN_DATA_POINTS:
            return None
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        if std < 0.001:
            return None
        return (current - mean) / std
