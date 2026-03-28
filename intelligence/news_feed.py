"""RSS news feed pipeline (Module 9).

Fetches RSS feeds every 60s, deduplicates by MD5 hash of title,
matches headlines to markets by keywords, and escalates to Groq
when VADER detects a significant sentiment shift.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import feedparser
import structlog

from config import settings
from intelligence.ai_analyzer import AIAnalyzer
from intelligence.sentiment import SentimentScorer

logger = structlog.get_logger(__name__)

# Default RSS feeds (configurable via config)
DEFAULT_RSS_FEEDS: dict[str, list[str]] = {
    "politics": [
        "https://feeds.reuters.com/Reuters/PoliticsNews",
        "https://rss.politico.com/politico.xml",
        "https://thehill.com/rss/syndicator/19109",
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss",
        "https://cointelegraph.com/rss",
    ],
    "general": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
    ],
}

FEED_POLL_INTERVAL_S = 60


@dataclass
class NewsItem:
    """A single news headline from RSS."""

    title: str
    link: str = ""
    source: str = ""
    category: str = ""
    hash: str = ""
    timestamp: float = 0.0


@dataclass
class MarketKeywords:
    """Keywords associated with a market for headline matching."""

    market_id: str
    question: str
    keywords: list[str] = field(default_factory=list)
    category: str = "other"
    probability: float = 0.0
    token_id: str = ""
    condition_id: str = ""


@dataclass
class NewsSignal:
    """Signal generated from a news headline matching a market."""

    market_id: str
    condition_id: str
    token_id: str
    headline: str
    direction: str  # "UP" or "DOWN"
    confidence: float
    should_trade: bool
    vader_score: float
    source: str  # "vader" or "groq"
    category: str = "other"


class NewsFeedPipeline:
    """Fetches RSS feeds, matches to markets, generates sentiment signals.

    Flow:
        1. Fetch RSS feeds every 60s
        2. Deduplicate by MD5 hash of title
        3. Match headlines to markets by keywords
        4. VADER score each match
        5. If shift > SENTIMENT_SHIFT, escalate to Groq
        6. Generate NewsSignal if actionable
    """

    def __init__(
        self,
        ai_analyzer: AIAnalyzer,
        feeds: dict[str, list[str]] | None = None,
    ) -> None:
        self._ai = ai_analyzer
        self._scorer = SentimentScorer()
        self._feeds = feeds or DEFAULT_RSS_FEEDS
        self._seen_hashes: set[str] = set()
        self._market_scores: dict[str, float] = {}  # market_id -> last VADER score
        self._market_keywords: dict[str, MarketKeywords] = {}
        self._running = False
        self._signals: list[NewsSignal] = []
        self._session: aiohttp.ClientSession | None = None

    def register_markets(self, markets: list[MarketKeywords]) -> None:
        """Register markets with their keywords for headline matching."""
        self._market_keywords = {m.market_id: m for m in markets}
        logger.info("news_feed_markets_registered", count=len(markets))

    def drain_signals(self) -> list[NewsSignal]:
        """Return and clear accumulated signals since last drain."""
        signals = self._signals.copy()
        self._signals.clear()
        return signals

    async def start(self) -> None:
        """Run the feed polling loop."""
        self._running = True
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        logger.info("news_feed_started", feeds=sum(len(v) for v in self._feeds.values()))

        try:
            while self._running:
                try:
                    await self._poll_feeds()
                except Exception as exc:
                    logger.error("news_feed_poll_error", error=str(exc))
                await asyncio.sleep(FEED_POLL_INTERVAL_S)
        finally:
            if self._session:
                await self._session.close()
                self._session = None

    async def stop(self) -> None:
        """Stop the feed polling loop."""
        self._running = False

    async def _poll_feeds(self) -> None:
        """Fetch all feeds and process new headlines."""
        all_urls = []
        for category, urls in self._feeds.items():
            for url in urls:
                all_urls.append((url, category))

        tasks = [self._fetch_feed(url, cat) for url, cat in all_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                new_items.extend(result)
            elif isinstance(result, Exception):
                logger.debug("feed_fetch_failed", error=str(result))

        if new_items:
            logger.info("news_items_fetched", new_count=len(new_items))

        for item in new_items:
            await self._process_item(item)

    async def _fetch_feed(self, url: str, category: str) -> list[NewsItem]:
        """Fetch and parse a single RSS feed."""
        if not self._session:
            return []

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return []
                content = await resp.text()
        except Exception:
            return []

        feed = feedparser.parse(content)
        items: list[NewsItem] = []

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue

            title_hash = hashlib.md5(title.encode()).hexdigest()
            if title_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(title_hash)

            items.append(
                NewsItem(
                    title=title,
                    link=entry.get("link", ""),
                    source=url,
                    category=category,
                    hash=title_hash,
                    timestamp=time.time(),
                )
            )

        return items

    async def _process_item(self, item: NewsItem) -> None:
        """Match a news item to markets and generate signals."""
        matched_markets = self._match_markets(item.title)

        for market in matched_markets:
            vader_score = self._scorer.compound_score(item.title)
            previous_score = self._market_scores.get(market.market_id, 0.0)
            shift = abs(vader_score - previous_score)
            self._market_scores[market.market_id] = vader_score

            if shift < settings.sentiment_shift:
                logger.debug(
                    "news_shift_below_threshold",
                    market_id=market.market_id,
                    shift=round(shift, 3),
                    threshold=settings.sentiment_shift,
                )
                continue

            # Escalate to Groq
            result = await self._ai.analyze_headline(
                item.title, market.question, market.probability
            )

            signal = NewsSignal(
                market_id=market.market_id,
                condition_id=market.condition_id,
                token_id=market.token_id,
                headline=item.title,
                direction=result.direction,
                confidence=result.confidence,
                should_trade=result.should_trade,
                vader_score=vader_score,
                source=result.source,
                category=market.category,
            )

            if signal.should_trade:
                self._signals.append(signal)
                logger.info(
                    "news_signal_generated",
                    market_id=market.market_id,
                    headline=item.title[:80],
                    direction=result.direction,
                    confidence=round(result.confidence, 3),
                    source=result.source,
                )

    def _match_markets(self, headline: str) -> list[MarketKeywords]:
        """Find markets where at least 1 keyword appears in the headline."""
        headline_lower = headline.lower()
        matched = []
        for market in self._market_keywords.values():
            if any(kw.lower() in headline_lower for kw in market.keywords):
                matched.append(market)
        return matched

    @staticmethod
    def extract_keywords(question: str) -> list[str]:
        """Extract meaningful keywords from a market question.

        Removes common stop words and returns lowercase keywords.
        """
        stop_words = {
            "will", "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "is", "be", "by", "it", "or", "and", "this", "that", "with",
            "from", "as", "are", "was", "were", "been", "being", "have",
            "has", "had", "do", "does", "did", "but", "not", "so", "if",
            "than", "too", "very", "can", "just", "should", "now", "before",
            "after", "during", "while", "where", "when", "how", "what",
            "which", "who", "whom", "why", "each", "every", "all", "any",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "only", "own", "same", "then", "about", "above", "below",
            "between", "into", "through", "up", "down", "out", "off",
            "over", "under", "again", "further", "once", "here", "there",
        }
        words = question.lower().replace("?", "").replace("!", "").split()
        return [w for w in words if w not in stop_words and len(w) > 2]
