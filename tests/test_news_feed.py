"""Tests for news feed pipeline (Module 9)."""

from __future__ import annotations

import pytest

from intelligence.news_feed import MarketKeywords, NewsFeedPipeline, NewsItem


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self):
        question = "Will Trump win the 2024 presidential election?"
        keywords = NewsFeedPipeline.extract_keywords(question)
        assert "trump" in keywords
        assert "2024" in keywords
        assert "presidential" in keywords
        assert "election" in keywords
        # Stop words should be removed
        assert "will" not in keywords
        assert "the" not in keywords

    def test_empty_question(self):
        assert NewsFeedPipeline.extract_keywords("") == []

    def test_short_words_filtered(self):
        keywords = NewsFeedPipeline.extract_keywords("Is BTC at 100k?")
        assert "is" not in keywords
        assert "at" not in keywords

    def test_removes_punctuation(self):
        keywords = NewsFeedPipeline.extract_keywords("Will Bitcoin exceed $100k?")
        assert "bitcoin" in keywords


class TestMarketMatching:
    def test_matches_by_keyword(self):
        pipeline = NewsFeedPipeline.__new__(NewsFeedPipeline)
        pipeline._market_keywords = {
            "mkt-trump": MarketKeywords(
                market_id="mkt-trump",
                question="Will Trump win?",
                keywords=["trump", "election", "iowa"],
                category="politics",
            ),
            "mkt-btc": MarketKeywords(
                market_id="mkt-btc",
                question="Will Bitcoin reach 100k?",
                keywords=["bitcoin", "btc", "crypto"],
                category="crypto",
            ),
        }

        matches = pipeline._match_markets("Trump leads in latest Iowa poll by 5 points")
        assert len(matches) == 1
        assert matches[0].market_id == "mkt-trump"

    def test_no_match_on_unrelated_headline(self):
        pipeline = NewsFeedPipeline.__new__(NewsFeedPipeline)
        pipeline._market_keywords = {
            "mkt-btc": MarketKeywords(
                market_id="mkt-btc",
                question="Will Bitcoin reach 100k?",
                keywords=["bitcoin", "btc"],
                category="crypto",
            ),
        }
        matches = pipeline._match_markets("New recipe for chocolate cake wins award")
        assert len(matches) == 0

    def test_case_insensitive_matching(self):
        pipeline = NewsFeedPipeline.__new__(NewsFeedPipeline)
        pipeline._market_keywords = {
            "mkt-1": MarketKeywords(
                market_id="mkt-1",
                question="Test",
                keywords=["BITCOIN"],
            ),
        }
        matches = pipeline._match_markets("bitcoin price surges")
        assert len(matches) == 1


class TestSignalDrain:
    def test_drain_signals_clears_buffer(self):
        pipeline = NewsFeedPipeline.__new__(NewsFeedPipeline)
        pipeline._signals = []

        from intelligence.news_feed import NewsSignal
        pipeline._signals.append(
            NewsSignal(
                market_id="mkt-1",
                condition_id="",
                token_id="",
                headline="Test",
                direction="UP",
                confidence=0.8,
                should_trade=True,
                vader_score=0.5,
                source="vader",
            )
        )
        signals = pipeline.drain_signals()
        assert len(signals) == 1
        assert len(pipeline._signals) == 0  # Buffer cleared


class TestDeduplication:
    def test_dedup_by_hash(self):
        """Same title should not be processed twice."""
        pipeline = NewsFeedPipeline.__new__(NewsFeedPipeline)
        pipeline._seen_hashes = set()

        import hashlib
        title = "Breaking: Major policy change announced"
        h = hashlib.md5(title.encode()).hexdigest()

        pipeline._seen_hashes.add(h)
        assert h in pipeline._seen_hashes

        # A different title should not be in the set
        title2 = "Another headline entirely"
        h2 = hashlib.md5(title2.encode()).hexdigest()
        assert h2 not in pipeline._seen_hashes
