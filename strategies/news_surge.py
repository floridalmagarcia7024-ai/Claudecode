"""News Sentiment Surge strategy.

Trades when VADER sentiment makes a strong directional move AND
volume is elevated — market hasn't yet priced in the news signal.

Entry conditions:
  - VADER compound score >= 0.30 (positive surge) or <= -0.30 (negative surge)
  - 24h volume >= 1.5x min_daily_volume (elevated activity)
  - Groq AI confirms direction (if available)
  - Probability not near resolution (0.05 < prob < 0.95)

Direction: positive sentiment → BUY, negative → SELL.
"""

from __future__ import annotations

import structlog

from config import settings
from core.state import PortfolioState
from intelligence.ai_analyzer import AIAnalyzer
from strategies.base import BaseStrategy, MarketContext, Signal

logger = structlog.get_logger(__name__)

SENTIMENT_THRESHOLD = 0.30
VOLUME_SURGE_MULTIPLIER = 1.5
RESOLUTION_GUARD = 0.05


class NewsSurgeStrategy(BaseStrategy):
    """Trade on news sentiment surges confirmed by volume."""

    def __init__(self, ai_analyzer: AIAnalyzer | None = None) -> None:
        self._ai = ai_analyzer

    @property
    def name(self) -> str:
        return "news_surge"

    async def generate_signal(self, context: MarketContext) -> Signal | None:
        market = context.market
        history = context.probability_history

        if self._ai is None:
            return None

        if len(history) < 6:
            return None

        prob = market.probability
        if prob < RESOLUTION_GUARD or prob > (1 - RESOLUTION_GUARD):
            return None

        # Volume gate: need elevated volume
        volume_threshold = settings.min_daily_volume * VOLUME_SURGE_MULTIPLIER
        if market.volume_24h < volume_threshold:
            logger.debug(
                "news_surge_rejected",
                market_id=market.market_id,
                reason="volume_insufficient",
                volume_24h=market.volume_24h,
                required=volume_threshold,
            )
            return None

        # Sentiment check
        compound = self._ai.analyze_sentiment(market.question)
        if abs(compound) < SENTIMENT_THRESHOLD:
            logger.debug(
                "news_surge_rejected",
                market_id=market.market_id,
                reason="sentiment_below_threshold",
                compound=round(compound, 3),
            )
            return None

        direction = "BUY" if compound > 0 else "SELL"

        # Optional Groq confirmation
        ai_confidence = None
        if abs(compound) > settings.sentiment_shift:
            try:
                groq_result = await self._ai.analyze_headline(
                    market.question, market.question, prob
                )
                ai_confidence = groq_result.confidence
                if not groq_result.should_trade:
                    logger.info(
                        "news_surge_rejected",
                        market_id=market.market_id,
                        reason="ai_no_trade",
                        ai_confidence=round(ai_confidence, 3),
                    )
                    return None
                # Directional mismatch check
                groq_dir = groq_result.direction
                if (direction == "BUY" and groq_dir == "DOWN") or (direction == "SELL" and groq_dir == "UP"):
                    logger.info(
                        "news_surge_rejected",
                        market_id=market.market_id,
                        reason="ai_direction_mismatch",
                        vader=direction,
                        groq=groq_dir,
                    )
                    return None
            except Exception:
                pass  # Groq unavailable — proceed with VADER only

        strength = min(abs(compound) / 0.8, 1.0)

        signal = Signal(
            market_id=market.market_id,
            condition_id=market.condition_id,
            token_id=market.token_ids[0] if market.token_ids else "",
            direction=direction,
            strength=strength,
            strategy=self.name,
            category=market.category,
            sentiment_score=compound,
            ai_confidence=ai_confidence,
            metadata={
                "compound": round(compound, 3),
                "volume_24h": market.volume_24h,
                "volume_threshold": volume_threshold,
            },
        )

        logger.info(
            "news_surge_signal",
            market_id=market.market_id,
            direction=direction,
            compound=round(compound, 3),
            strength=round(strength, 3),
        )
        return signal

    def calculate_size(self, signal: Signal, portfolio: PortfolioState) -> float:
        return portfolio.capital * (settings.default_position_pct / 100)
