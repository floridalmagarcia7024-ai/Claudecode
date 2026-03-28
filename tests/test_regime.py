"""Tests for market regime detector (Module 10)."""

from __future__ import annotations

import random

import pytest

from core.regime import MarketRegime, MarketRegimeDetector, MIN_DATA_POINTS_REGIME


@pytest.fixture
def detector() -> MarketRegimeDetector:
    return MarketRegimeDetector()


class TestRegimeDetection:
    def test_insufficient_data_defaults_ranging(self, detector: MarketRegimeDetector):
        history = [0.50] * 10  # Way below minimum
        regime = detector.detect_regime("mkt-1", history)
        assert regime == MarketRegime.RANGING

    def test_ranging_regime(self, detector: MarketRegimeDetector):
        """Flat price around 0.50 ±1% should be RANGING."""
        random.seed(42)
        # Generate tight oscillation around 0.50
        history = [0.50 + random.uniform(-0.005, 0.005) for _ in range(200)]
        regime = detector.detect_regime("mkt-ranging", history)
        assert regime == MarketRegime.RANGING

    def test_trending_up_regime(self, detector: MarketRegimeDetector):
        """Steadily increasing prices should be TRENDING_UP."""
        # Create clear upward trend: 0.40 to 0.60 over 200 points
        history = [0.40 + i * 0.001 for i in range(200)]
        regime = detector.detect_regime("mkt-up", history)
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down_regime(self, detector: MarketRegimeDetector):
        """Steadily decreasing prices should be TRENDING_DOWN."""
        history = [0.60 - i * 0.001 for i in range(200)]
        regime = detector.detect_regime("mkt-down", history)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_high_volatility_regime(self, detector: MarketRegimeDetector):
        """ATR > 2x 30-day average should trigger HIGH_VOLATILITY."""
        history = [0.50] * 200  # Doesn't matter much
        # 30 days of ATR, last value is 3x the average
        atr_history = [0.01] * 29 + [0.03]
        regime = detector.detect_regime("mkt-vol", history, atr_history_30d=atr_history)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_high_volatility_not_triggered_normal_atr(self, detector: MarketRegimeDetector):
        """Normal ATR should not trigger HIGH_VOLATILITY."""
        history = [0.50 + random.uniform(-0.005, 0.005) for _ in range(200)]
        atr_history = [0.01] * 30  # All the same
        regime = detector.detect_regime("mkt-normal", history, atr_history_30d=atr_history)
        assert regime != MarketRegime.HIGH_VOLATILITY

    def test_regime_caching(self, detector: MarketRegimeDetector):
        """Regime should be cached after detection."""
        history = [0.50] * 200
        detector.detect_regime("mkt-cache", history)
        assert detector.get_regime("mkt-cache") == MarketRegime.RANGING

    def test_default_regime_for_unknown_market(self, detector: MarketRegimeDetector):
        assert detector.get_regime("unknown") == MarketRegime.RANGING


class TestATRComputation:
    def test_atr_computation(self):
        closes = [10.0 + i * 0.1 for i in range(30)]
        atrs = MarketRegimeDetector.compute_atr(closes, period=14)
        assert len(atrs) > 0
        assert all(a >= 0 for a in atrs)

    def test_atr_insufficient_data(self):
        closes = [10.0] * 5
        atrs = MarketRegimeDetector.compute_atr(closes, period=14)
        assert atrs == []

    def test_atr_constant_price(self):
        closes = [10.0] * 30
        atrs = MarketRegimeDetector.compute_atr(closes, period=14)
        assert all(a == pytest.approx(0.0, abs=1e-10) for a in atrs)
