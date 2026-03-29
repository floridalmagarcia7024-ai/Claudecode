"""Cross-market intelligence — Metaculus + Manifold divergences (Module 24).

Fetches prediction probabilities from external platforms and compares
them with Polymarket prices. A divergence > 8 points generates a signal.

External APIs (public, no auth required):
  Metaculus: api.metaculus.com/api2/questions/
  Manifold: manifold.markets/api/v0/markets

Limitation: Keyword matching between platforms is a rough approximation.
Semantic question matching is an NLP-hard problem — this module uses
keyword overlap as a first approximation and does not assume perfection.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

DIVERGENCE_THRESHOLD = 0.08  # 8 percentage points
FETCH_TIMEOUT = 15
MAX_EXTERNAL_MARKETS = 200

# Stop words for keyword extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "could", "should", "may", "might", "can", "do", "does",
    "did", "has", "have", "had", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "or", "and", "but", "if", "then", "than", "that",
    "this", "it", "its", "not", "no", "yes", "what", "which", "who", "whom",
    "how", "when", "where", "why", "before", "after", "during", "between",
    "through", "about", "into", "over", "under",
}


@dataclass
class ExternalMarket:
    """A prediction market from an external platform."""

    platform: str  # "metaculus" or "manifold"
    question: str
    probability: float  # 0.0 - 1.0
    url: str = ""
    volume: float = 0.0
    keywords: set[str] = field(default_factory=set)


@dataclass
class DivergenceSignal:
    """Signal generated when Polymarket diverges from external sources."""

    polymarket_id: str
    polymarket_question: str
    polymarket_prob: float
    external_platform: str
    external_question: str
    external_prob: float
    divergence: float  # signed: poly - external
    abs_divergence: float
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "polymarket_id": self.polymarket_id,
            "polymarket_question": self.polymarket_question[:80],
            "polymarket_prob": round(self.polymarket_prob, 3),
            "external_platform": self.external_platform,
            "external_question": self.external_question[:80],
            "external_prob": round(self.external_prob, 3),
            "divergence": round(self.divergence, 3),
            "abs_divergence": round(self.abs_divergence, 3),
            "timestamp": self.timestamp,
        }


class CrossMarketIntelligence:
    """Compares Polymarket prices with Metaculus and Manifold predictions.

    For each active Polymarket market, searches for matching questions
    on external platforms using keyword overlap. If divergence > 8 points,
    generates a DivergenceSignal.
    """

    def __init__(self) -> None:
        self._external_cache: list[ExternalMarket] = []
        self._last_fetch: float = 0
        self._cache_ttl: float = 3600  # Refresh external data every hour
        self._signals: list[DivergenceSignal] = []
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from a question."""
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {w for w in words if w not in STOP_WORDS}

    @staticmethod
    def keyword_similarity(kw1: set[str], kw2: set[str]) -> float:
        """Jaccard similarity between two keyword sets."""
        if not kw1 or not kw2:
            return 0.0
        intersection = kw1 & kw2
        union = kw1 | kw2
        return len(intersection) / len(union)

    async def fetch_metaculus(self) -> list[ExternalMarket]:
        """Fetch active binary questions from Metaculus."""
        markets = []
        try:
            session = await self._get_session()
            url = "https://api.metaculus.com/api2/questions/"
            params = {
                "status": "open",
                "type": "binary",
                "limit": 100,
                "order_by": "-activity",
            }
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("metaculus_fetch_failed", status=resp.status)
                    return []
                data = await resp.json()

            results = data.get("results", [])
            for q in results[:MAX_EXTERNAL_MARKETS]:
                title = q.get("title", "")
                prediction = q.get("community_prediction", {})
                prob = prediction.get("full", {}).get("q2") if isinstance(prediction, dict) else None
                if prob is None:
                    continue

                markets.append(ExternalMarket(
                    platform="metaculus",
                    question=title,
                    probability=float(prob),
                    url=f"https://www.metaculus.com/questions/{q.get('id', '')}/",
                    keywords=self.extract_keywords(title),
                ))

            logger.info("metaculus_fetched", count=len(markets))
        except Exception as exc:
            logger.warning("metaculus_error", error=str(exc))

        return markets

    async def fetch_manifold(self) -> list[ExternalMarket]:
        """Fetch active binary markets from Manifold."""
        markets = []
        try:
            session = await self._get_session()
            url = "https://api.manifold.markets/v0/markets"
            params = {"limit": 100, "sort": "liquidity"}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("manifold_fetch_failed", status=resp.status)
                    return []
                data = await resp.json()

            for m in data[:MAX_EXTERNAL_MARKETS]:
                if m.get("outcomeType") != "BINARY":
                    continue
                if m.get("isResolved", False):
                    continue

                title = m.get("question", "")
                prob = m.get("probability", 0)
                volume = m.get("totalLiquidity", 0)

                markets.append(ExternalMarket(
                    platform="manifold",
                    question=title,
                    probability=float(prob),
                    url=m.get("url", ""),
                    volume=float(volume),
                    keywords=self.extract_keywords(title),
                ))

            logger.info("manifold_fetched", count=len(markets))
        except Exception as exc:
            logger.warning("manifold_error", error=str(exc))

        return markets

    async def refresh_external_data(self) -> None:
        """Fetch data from all external platforms."""
        if time.time() - self._last_fetch < self._cache_ttl:
            return

        metaculus, manifold = await asyncio.gather(
            self.fetch_metaculus(),
            self.fetch_manifold(),
            return_exceptions=True,
        )

        self._external_cache = []
        if isinstance(metaculus, list):
            self._external_cache.extend(metaculus)
        if isinstance(manifold, list):
            self._external_cache.extend(manifold)

        self._last_fetch = time.time()
        logger.info("external_data_refreshed", total=len(self._external_cache))

    def find_matches(
        self, polymarket_question: str, min_similarity: float = 0.25
    ) -> list[ExternalMarket]:
        """Find external markets matching a Polymarket question by keywords."""
        poly_kw = self.extract_keywords(polymarket_question)
        if not poly_kw:
            return []

        matches = []
        for ext in self._external_cache:
            sim = self.keyword_similarity(poly_kw, ext.keywords)
            if sim >= min_similarity:
                matches.append(ext)

        # Sort by similarity (best first)
        matches.sort(
            key=lambda m: self.keyword_similarity(poly_kw, m.keywords),
            reverse=True,
        )
        return matches[:3]  # Top 3 matches

    async def check_divergences(
        self, polymarket_markets: list[dict]
    ) -> list[DivergenceSignal]:
        """Check all Polymarket markets for divergences with external sources.

        Args:
            polymarket_markets: List of dicts with market_id, question, probability.

        Returns:
            List of divergence signals.
        """
        await self.refresh_external_data()

        new_signals = []
        now = datetime.now(timezone.utc).isoformat()

        for pm in polymarket_markets:
            question = pm.get("question", "")
            prob = pm.get("probability", 0.5)
            market_id = pm.get("market_id", "")

            matches = self.find_matches(question)
            for ext in matches:
                divergence = prob - ext.probability
                abs_div = abs(divergence)

                if abs_div >= DIVERGENCE_THRESHOLD:
                    signal = DivergenceSignal(
                        polymarket_id=market_id,
                        polymarket_question=question,
                        polymarket_prob=prob,
                        external_platform=ext.platform,
                        external_question=ext.question,
                        external_prob=ext.probability,
                        divergence=divergence,
                        abs_divergence=abs_div,
                        timestamp=now,
                    )
                    new_signals.append(signal)
                    logger.info(
                        "divergence_detected",
                        market_id=market_id[:30],
                        platform=ext.platform,
                        divergence=round(abs_div, 3),
                    )

        self._signals = new_signals
        return new_signals

    def get_recent_signals(self) -> list[dict]:
        """Get recent divergence signals for dashboard."""
        return [s.to_dict() for s in self._signals[-20:]]

    def drain_signals(self) -> list[DivergenceSignal]:
        """Drain and return all pending signals."""
        signals = self._signals.copy()
        self._signals.clear()
        return signals
