"""AI-powered trade journal analysis using Groq (Module 20).

Post-close analysis: 3 bullets per trade:
  - What went well
  - What went wrong
  - What to adjust
"""

from __future__ import annotations

import json

import structlog

from config import settings

logger = structlog.get_logger(__name__)


class AIJournalAnalyzer:
    """Generates post-trade analysis using Groq LLM."""

    def __init__(self) -> None:
        self._groq = None
        if settings.groq_api_key:
            try:
                from groq import AsyncGroq
                self._groq = AsyncGroq(api_key=settings.groq_api_key)
            except ImportError:
                logger.warning("groq_not_available_for_journal")

    @property
    def available(self) -> bool:
        return self._groq is not None

    async def analyze_trade(
        self,
        market_id: str,
        strategy: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        regime: str = "",
        related_news: str = "",
    ) -> str:
        """Generate 3-bullet post-trade analysis.

        Returns:
            Analysis text or empty string if unavailable.
        """
        if not self._groq:
            return ""

        prompt = (
            "You are a trading analyst. Analyze this completed trade and provide "
            "exactly 3 bullet points:\n"
            "1. What went well\n"
            "2. What went wrong\n"
            "3. What to adjust next time\n\n"
            f"Market: {market_id}\n"
            f"Strategy: {strategy}\n"
            f"Direction: {direction}\n"
            f"Entry: {entry_price:.4f} -> Exit: {exit_price:.4f}\n"
            f"PnL: ${pnl_usd:.2f}\n"
            f"Exit reason: {exit_reason}\n"
            f"Regime: {regime}\n"
            f"Related news: {related_news[:200] if related_news else 'None'}\n\n"
            "Respond with exactly 3 bullet points, each starting with a bullet character. "
            "Keep each point under 50 words."
        )

        try:
            response = await self._groq.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
            logger.info("ai_journal_analysis_complete", market_id=market_id)
            return content.strip()
        except Exception as exc:
            logger.error("ai_journal_analysis_failed", error=str(exc))
            return ""
