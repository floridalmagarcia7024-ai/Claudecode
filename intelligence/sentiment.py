"""Clean VADER sentiment wrapper for reuse across modules.

Provides a simple interface for sentiment scoring used by
news_feed, mean_reversion, and other components.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentScorer:
    """Thin wrapper around VADER for consistent sentiment scoring.

    VADER only works with English text.
    """

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()

    def compound_score(self, text: str) -> float:
        """Return VADER compound score in [-1.0, 1.0]."""
        return self._vader.polarity_scores(text)["compound"]

    def score_shift(self, text: str, previous_score: float) -> float:
        """Return absolute shift between current and previous score."""
        return abs(self.compound_score(text) - previous_score)
