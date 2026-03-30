"""Value Bet strategy — trade when AI probability diverges from market price.

Edge = AI_estimated_probability - market_probability

Logic:
  - AI_prob >> market_prob (edge > +10%): market undervaluing YES → BUY
  - AI_prob << market_prob (edge < -10%): market overvaluing YES → SELL

Requires Groq API key. Skips signal if Groq unavailable.
Only trades in liquid markets with probability between 10% and 90%.
"""

from __future__ import annotations

import structlog

from config import settings
from core.state import PortfolioState
from intelligence.ai_analyzer import AIAnalyzer
from strategies.base import BaseStrategy, MarketContext, Signal

logger = structlog.get_logger(__name__)

EDGE_THRESHOLD = 0.10
RESOLUTION_GUARD_LOW = 0.10
RESOLUTION_GUARD_HIGH = 0.90


class ValueBetStrategy(BaseStrategy):
    """Trade when AI-estimated probability diverges from market price."""

    def __init__(self, ai_analyzer: AIAnalyzer | None = None) -> None:
        self._ai = ai_analyzer

    @property
    def name(self) -> str:
        return "value_bet"

    async def generate_signal(self, context: MarketContext) -> Signal | None:
        market = context.market

        if self._ai is None:
            return None

        prob = market.probability
        if prob < RESOLUTION_GUARD_LOW or prob > RESOLUTION_GUARD_HIGH:
            return None

        if market.volume_24h < settings.min_daily_volume:
            return None

        # Get AI probability estimate
        try:
            groq_result = await self._ai.analyze_headline(
                market.question, market.question, prob
            )
        except Exception:
            return None  # Groq unavailable

        ai_confidence = groq_result.confidence

        # Convert direction + confidence to probability estimate
        if groq_result.direction == "UP":
            ai_prob = 0.5 + ai_confidence * 0.5
        elif groq_result.direction == "DOWN":
            ai_prob = 0.5 - ai_confidence * 0.5
        else:
            return None  # NEUTRAL — no actionable edge

        edge = ai_prob - prob

        if abs(edge) < EDGE_THRESHOLD:
            logger.debug(
                "value_bet_rejected",
                market_id=market.market_id,
                reason="edge_below_threshold",
                edge=round(edge, 3),
                threshold=EDGE_THRESHOLD,
                ai_prob=round(ai_prob, 3),
                market_prob=round(prob, 3),
            )
            return None

        direction = "BUY" if edge > 0 else "SELL"
        strength = min(abs(edge) / 0.30, 1.0)

        signal = Signal(
            market_id=market.market_id,
            condition_id=market.condition_id,
            token_id=market.token_ids[0] if market.token_ids else "",
            direction=direction,
            strength=strength,
            strategy=self.name,
            category=market.category,
            ai_confidence=ai_confidence,
            metadata={
                "ai_prob": round(ai_prob, 3),
                "market_prob": round(prob, 3),
                "edge": round(edge, 3),
                "volume_24h": market.volume_24h,
            },
        )

        logger.info(
            "value_bet_signal",
            market_id=market.market_id,
            direction=direction,
            edge=round(edge, 3),
            ai_prob=round(ai_prob, 3),
            market_prob=round(prob, 3),
        )
        return signal

    def calculate_size(self, signal: Signal, portfolio: PortfolioState) -> float:
        # Scale size proportionally to edge (Kelly-inspired)
        edge = abs(signal.metadata.get("edge", EDGE_THRESHOLD))
        fraction = min(edge * 1.5, settings.default_position_pct / 100)
        return portfolio.capital * fraction
