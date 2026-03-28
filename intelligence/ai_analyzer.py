"""AI analysis layer: VADER (offline) + Groq (free tier).

VADER handles instant sentiment scoring (English only).
Groq provides deep analysis when VADER detects a significant shift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import structlog
from groq import AsyncGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import settings

logger = structlog.get_logger(__name__)


@dataclass
class AnalysisResult:
    """Result from AI analysis."""

    direction: Literal["UP", "DOWN", "NEUTRAL"]
    confidence: float  # 0.0 - 1.0 (NOT a calibrated probability)
    impact: str  # "high", "medium", "low"
    should_trade: bool
    reasoning: str = ""
    source: str = ""  # "vader", "groq", "vader_fallback"


class AIAnalyzer:
    """Dual AI stack: VADER for quick sentiment, Groq for deep analysis.

    IMPORTANT: VADER only works with English text.
    Groq is rate-limited to a conservative daily limit (300 of 14,400 free).
    """

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._groq: AsyncGroq | None = None
        if settings.groq_api_key:
            self._groq = AsyncGroq(api_key=settings.groq_api_key)

        self._daily_limit = settings.groq_daily_limit
        self._calls_today = 0
        self._current_date = datetime.now(timezone.utc).date()

    def _reset_daily_counter_if_needed(self) -> None:
        """Reset Groq call counter at UTC midnight."""
        today = datetime.now(timezone.utc).date()
        if today != self._current_date:
            self._calls_today = 0
            self._current_date = today
            logger.info("groq_counter_reset", date=str(today))

    @property
    def groq_available(self) -> bool:
        """Check if Groq calls are available today."""
        self._reset_daily_counter_if_needed()
        return self._groq is not None and self._calls_today < self._daily_limit

    @property
    def groq_calls_remaining(self) -> int:
        """Number of Groq calls remaining today."""
        self._reset_daily_counter_if_needed()
        return max(0, self._daily_limit - self._calls_today)

    def analyze_sentiment(self, text: str) -> float:
        """Get VADER compound sentiment score for text.

        Args:
            text: English text to analyze.

        Returns:
            Compound score in [-1.0, 1.0].
        """
        scores = self._vader.polarity_scores(text)
        return scores["compound"]

    def vader_analysis(self, text: str) -> AnalysisResult:
        """Quick VADER-only analysis.

        Direction rules:
            compound > 0.1  → UP
            compound < -0.1 → DOWN
            else            → NEUTRAL

        should_trade = abs(compound) > 0.30
        """
        score = self.analyze_sentiment(text)
        abs_score = abs(score)

        if score > 0.1:
            direction: Literal["UP", "DOWN", "NEUTRAL"] = "UP"
        elif score < -0.1:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        should_trade = abs_score > 0.30

        return AnalysisResult(
            direction=direction,
            confidence=min(abs_score, 1.0),
            impact="medium" if abs_score > 0.5 else "low",
            should_trade=should_trade,
            reasoning=f"VADER compound={score:.3f}",
            source="vader",
        )

    async def analyze_with_groq(
        self, text: str, market_question: str, current_probability: float
    ) -> AnalysisResult:
        """Deep analysis using Groq LLM.

        Only called when VADER detects shift > SENTIMENT_SHIFT.
        Falls back to VADER if Groq is unavailable.

        Args:
            text: News headline or text to analyze (English).
            market_question: The market's question for context.
            current_probability: Current market probability.

        Returns:
            AnalysisResult from Groq or VADER fallback.
        """
        if not self.groq_available:
            logger.info(
                "groq_unavailable_fallback",
                calls_today=self._calls_today,
                limit=self._daily_limit,
            )
            result = self.vader_analysis(text)
            result.source = "vader_fallback"
            return result

        prompt = (
            "You are a prediction market analyst. Analyze how this news affects "
            "the probability of the following market.\n\n"
            f"Market: {market_question}\n"
            f"Current probability: {current_probability:.1%}\n"
            f"News: {text}\n\n"
            "Respond ONLY with valid JSON (no markdown):\n"
            '{"direction": "UP" or "DOWN" or "NEUTRAL", '
            '"confidence": 0.0-1.0, '
            '"impact": "high" or "medium" or "low", '
            '"should_trade": true/false, '
            '"reasoning": "brief explanation"}'
        )

        try:
            self._calls_today += 1
            response = await self._groq.chat.completions.create(  # type: ignore[union-attr]
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            content = response.choices[0].message.content or ""
            data = json.loads(content)

            confidence = float(data.get("confidence", 0.0))
            should_trade = bool(data.get("should_trade", False))

            # Override: if confidence below minimum, don't trade
            if confidence < settings.ai_confidence_min:
                should_trade = False

            result = AnalysisResult(
                direction=data.get("direction", "NEUTRAL"),
                confidence=confidence,
                impact=data.get("impact", "low"),
                should_trade=should_trade,
                reasoning=data.get("reasoning", ""),
                source="groq",
            )

            logger.info(
                "groq_analysis_complete",
                direction=result.direction,
                confidence=result.confidence,
                should_trade=result.should_trade,
                calls_remaining=self.groq_calls_remaining,
            )
            return result

        except json.JSONDecodeError as exc:
            logger.warning("groq_json_parse_error", error=str(exc))
            result = self.vader_analysis(text)
            result.source = "vader_fallback"
            return result
        except Exception as exc:
            logger.error("groq_api_error", error=str(exc))
            result = self.vader_analysis(text)
            result.source = "vader_fallback"
            return result

    async def analyze_headline(
        self,
        headline: str,
        market_question: str,
        current_probability: float,
    ) -> AnalysisResult:
        """Main entry point: analyze a headline with escalation logic.

        1. Always run VADER first.
        2. If VADER shift > SENTIMENT_SHIFT, escalate to Groq.
        3. Otherwise, return VADER result.

        Args:
            headline: English news headline.
            market_question: Market question for context.
            current_probability: Current market probability.

        Returns:
            AnalysisResult from the appropriate source.
        """
        vader_score = self.analyze_sentiment(headline)
        abs_score = abs(vader_score)

        logger.debug(
            "vader_score",
            headline=headline[:80],
            score=round(vader_score, 3),
        )

        # Escalate to Groq if shift is significant
        if abs_score > settings.sentiment_shift and self.groq_available:
            logger.info(
                "escalating_to_groq",
                vader_score=round(vader_score, 3),
                threshold=settings.sentiment_shift,
            )
            return await self.analyze_with_groq(
                headline, market_question, current_probability
            )

        return self.vader_analysis(headline)
