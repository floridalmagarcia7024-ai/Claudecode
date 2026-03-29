"""Tests for CrossMarketIntelligence (Module 24)."""

import pytest

from intelligence.cross_market import (
    CrossMarketIntelligence,
    DivergenceSignal,
    ExternalMarket,
    DIVERGENCE_THRESHOLD,
)


class TestCrossMarketIntelligence:
    def setup_method(self):
        self.intel = CrossMarketIntelligence()

    def test_extract_keywords(self):
        kw = CrossMarketIntelligence.extract_keywords(
            "Will Trump win the 2024 presidential election?"
        )
        assert "trump" in kw
        assert "presidential" in kw
        assert "election" in kw
        # Stop words excluded
        assert "will" not in kw
        assert "the" not in kw

    def test_keyword_similarity_identical(self):
        kw = {"trump", "election", "president"}
        sim = CrossMarketIntelligence.keyword_similarity(kw, kw)
        assert sim == 1.0

    def test_keyword_similarity_no_overlap(self):
        kw1 = {"bitcoin", "crypto"}
        kw2 = {"trump", "election"}
        sim = CrossMarketIntelligence.keyword_similarity(kw1, kw2)
        assert sim == 0.0

    def test_keyword_similarity_partial(self):
        kw1 = {"trump", "election", "win"}
        kw2 = {"trump", "election", "president", "2024"}
        sim = CrossMarketIntelligence.keyword_similarity(kw1, kw2)
        assert 0.0 < sim < 1.0

    def test_keyword_similarity_empty(self):
        assert CrossMarketIntelligence.keyword_similarity(set(), {"a"}) == 0.0
        assert CrossMarketIntelligence.keyword_similarity(set(), set()) == 0.0

    def test_find_matches_with_cache(self):
        self.intel._external_cache = [
            ExternalMarket(
                platform="metaculus",
                question="Will Trump win 2024 election?",
                probability=0.55,
                keywords={"trump", "win", "2024", "election"},
            ),
            ExternalMarket(
                platform="manifold",
                question="Bitcoin price above 100k by 2025?",
                probability=0.30,
                keywords={"bitcoin", "price", "above", "100k", "2025"},
            ),
        ]
        matches = self.intel.find_matches("Will Trump win the presidential election in 2024?")
        assert len(matches) >= 1
        assert matches[0].platform == "metaculus"

    def test_find_matches_no_match(self):
        self.intel._external_cache = [
            ExternalMarket(
                platform="metaculus",
                question="Will it rain tomorrow?",
                probability=0.5,
                keywords={"rain", "tomorrow"},
            ),
        ]
        matches = self.intel.find_matches("Will Bitcoin reach 100k?")
        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_check_divergences_generates_signal(self):
        self.intel._external_cache = [
            ExternalMarket(
                platform="manifold",
                question="Trump wins 2024 election",
                probability=0.40,
                keywords={"trump", "wins", "2024", "election"},
            ),
        ]
        self.intel._last_fetch = 9999999999  # Skip refresh

        pm_markets = [
            {
                "market_id": "pm_1",
                "question": "Will Trump win the 2024 election?",
                "probability": 0.55,  # Divergence = 0.15 > 0.08 threshold
            }
        ]
        signals = await self.intel.check_divergences(pm_markets)
        assert len(signals) >= 1
        assert signals[0].abs_divergence >= DIVERGENCE_THRESHOLD

    @pytest.mark.asyncio
    async def test_check_divergences_no_signal_small_diff(self):
        self.intel._external_cache = [
            ExternalMarket(
                platform="manifold",
                question="Trump wins 2024 election",
                probability=0.52,
                keywords={"trump", "wins", "2024", "election"},
            ),
        ]
        self.intel._last_fetch = 9999999999

        pm_markets = [
            {
                "market_id": "pm_1",
                "question": "Will Trump win the 2024 election?",
                "probability": 0.55,  # Divergence = 0.03 < 0.08
            }
        ]
        signals = await self.intel.check_divergences(pm_markets)
        assert len(signals) == 0

    def test_get_recent_signals(self):
        assert self.intel.get_recent_signals() == []

    def test_drain_signals(self):
        self.intel._signals = [
            DivergenceSignal(
                polymarket_id="pm_1",
                polymarket_question="Q",
                polymarket_prob=0.5,
                external_platform="manifold",
                external_question="Q ext",
                external_prob=0.6,
                divergence=-0.1,
                abs_divergence=0.1,
            )
        ]
        drained = self.intel.drain_signals()
        assert len(drained) == 1
        assert len(self.intel._signals) == 0

    def test_divergence_signal_to_dict(self):
        sig = DivergenceSignal(
            polymarket_id="pm_1",
            polymarket_question="Question?",
            polymarket_prob=0.55,
            external_platform="metaculus",
            external_question="External Q?",
            external_prob=0.40,
            divergence=0.15,
            abs_divergence=0.15,
            timestamp="2024-01-01T00:00:00",
        )
        d = sig.to_dict()
        assert d["abs_divergence"] == 0.15
        assert d["external_platform"] == "metaculus"
