"""Tests for ShadowBotManager (Module 22)."""

import pytest

from optimizer.shadow_bot import ShadowBotManager, MAX_SHADOW_BOTS


class TestShadowBotManager:
    def setup_method(self):
        self.manager = ShadowBotManager()

    def test_create_shadow_bot(self):
        result = self.manager.create_shadow_bot(
            "shadow_1", {"zscore_threshold": 2.0}
        )
        assert result is True
        assert "shadow_1" in self.manager.bots

    def test_create_max_bots(self):
        self.manager.create_shadow_bot("s1", {"zscore_threshold": 1.5})
        self.manager.create_shadow_bot("s2", {"zscore_threshold": 2.5})
        result = self.manager.create_shadow_bot("s3", {"zscore_threshold": 2.0})
        assert result is False
        assert len(self.manager.bots) == MAX_SHADOW_BOTS

    def test_create_duplicate_rejected(self):
        self.manager.create_shadow_bot("s1", {"zscore_threshold": 1.5})
        result = self.manager.create_shadow_bot("s1", {"zscore_threshold": 2.0})
        assert result is False

    def test_remove_shadow_bot(self):
        self.manager.create_shadow_bot("s1", {})
        assert self.manager.remove_shadow_bot("s1") is True
        assert "s1" not in self.manager.bots

    def test_remove_nonexistent(self):
        assert self.manager.remove_shadow_bot("nope") is False

    def test_evaluate_signal_opens_trade(self):
        self.manager.create_shadow_bot(
            "s1",
            {"zscore_threshold": 1.5, "sentiment_shift": 0.10, "ai_confidence_min": 0.4},
        )
        result = self.manager.evaluate_signal(
            bot_id="s1",
            market_id="market_1",
            direction="BUY",
            price=0.55,
            size_usd=100,
            strategy="mean_reversion",
            z_score=-2.0,
            sentiment=0.30,
            ai_confidence=0.7,
        )
        assert result is True
        assert len(self.manager.bots["s1"].trades) == 1
        assert self.manager.bots["s1"].trades[0].is_open

    def test_evaluate_signal_rejects_low_zscore(self):
        self.manager.create_shadow_bot(
            "s1", {"zscore_threshold": 2.5, "sentiment_shift": 0.1, "ai_confidence_min": 0.4}
        )
        result = self.manager.evaluate_signal(
            bot_id="s1",
            market_id="m1",
            direction="BUY",
            price=0.55,
            size_usd=100,
            strategy="mean_reversion",
            z_score=-1.5,  # Below 2.5 threshold
            sentiment=0.30,
            ai_confidence=0.7,
        )
        assert result is False

    def test_close_shadow_trade(self):
        self.manager.create_shadow_bot(
            "s1", {"zscore_threshold": 1.0, "sentiment_shift": 0.05, "ai_confidence_min": 0.3}
        )
        self.manager.evaluate_signal(
            "s1", "m1", "BUY", 0.50, 100, "mean_reversion",
            z_score=-2.0, sentiment=0.3, ai_confidence=0.7,
        )
        trade = self.manager.close_shadow_trade("s1", "m1", 0.60)
        assert trade is not None
        assert not trade.is_open
        assert trade.pnl_usd > 0  # Profitable BUY

    def test_close_nonexistent_trade(self):
        self.manager.create_shadow_bot("s1", {})
        trade = self.manager.close_shadow_trade("s1", "nonexistent", 0.5)
        assert trade is None

    def test_get_performance(self):
        self.manager.create_shadow_bot("s1", {"zscore_threshold": 2.0})
        perf = self.manager.get_performance()
        assert len(perf) == 1
        assert perf[0]["bot_id"] == "s1"
        assert perf[0]["total_pnl"] == 0.0

    def test_check_graduation_too_young(self):
        self.manager.create_shadow_bot("s1", {})
        assert not self.manager.check_graduation("s1", real_pnl=0.0)

    def test_update_open_positions(self):
        self.manager.create_shadow_bot(
            "s1", {"zscore_threshold": 1.0, "sentiment_shift": 0.05, "ai_confidence_min": 0.3}
        )
        self.manager.evaluate_signal(
            "s1", "m1", "BUY", 0.50, 100, "mean_reversion",
            z_score=-2.0, sentiment=0.3, ai_confidence=0.7,
        )
        self.manager.update_open_positions({"m1": 0.55})
        trade = self.manager.bots["s1"].trades[0]
        assert trade.pnl_usd > 0
