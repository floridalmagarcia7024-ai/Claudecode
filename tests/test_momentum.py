"""Tests for momentum strategy (Module 11)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from api.polymarket import MarketData
from core.regime import MarketRegime, MarketRegimeDetector
from core.state import PortfolioState
from strategies.base import MarketContext
from strategies.momentum import (
    CONVERGENCE_THRESHOLD,
    MOMENTUM_CHANGE_THRESHOLD,
    MOMENTUM_EXIT_THRESHOLD,
    MomentumStrategy,
)


class MockRegimeDetector:
    """Mock regime detector for testing."""

    def __init__(self, regime: MarketRegime = MarketRegime.TRENDING_UP):
        self._regime = regime
        self._regimes: dict[str, MarketRegime] = {}

    def get_regime(self, market_id: str) -> MarketRegime:
        return self._regimes.get(market_id, self._regime)

    def set_regime(self, market_id: str, regime: MarketRegime) -> None:
        self._regimes[market_id] = regime


@pytest.fixture
def trending_up_strategy() -> MomentumStrategy:
    return MomentumStrategy(regime_detector=MockRegimeDetector(MarketRegime.TRENDING_UP))


@pytest.fixture
def ranging_strategy() -> MomentumStrategy:
    return MomentumStrategy(regime_detector=MockRegimeDetector(MarketRegime.RANGING))


class TestMomentumSignals:
    @pytest.mark.asyncio
    async def test_buy_signal_on_upward_momentum(self, trending_up_strategy: MomentumStrategy):
        """Strong upward move in trending market should generate BUY."""
        # History: price was 0.50, now at 0.55 (change = 0.05 > 0.025)
        history = [0.50] * 12 + [0.55]
        market = MarketData(
            market_id="mkt-mom",
            condition_id="cond",
            probability=0.55,
            volume_24h=5000,
            token_ids=["tok-1"],
            category="crypto",
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await trending_up_strategy.generate_signal(context)
        assert signal is not None
        assert signal.direction == "BUY"
        assert signal.strategy == "momentum"

    @pytest.mark.asyncio
    async def test_sell_signal_on_downward_momentum(self):
        """Strong downward move in trending down market should generate SELL."""
        strategy = MomentumStrategy(
            regime_detector=MockRegimeDetector(MarketRegime.TRENDING_DOWN)
        )
        history = [0.55] * 12 + [0.50]
        market = MarketData(
            market_id="mkt-mom-down",
            condition_id="cond",
            probability=0.50,
            volume_24h=5000,
            token_ids=["tok-1"],
            category="crypto",
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await strategy.generate_signal(context)
        assert signal is not None
        assert signal.direction == "SELL"

    @pytest.mark.asyncio
    async def test_no_signal_in_ranging_regime(self, ranging_strategy: MomentumStrategy):
        """Momentum should not fire in RANGING regime."""
        history = [0.50] * 12 + [0.55]
        market = MarketData(
            market_id="mkt-range",
            probability=0.55,
            volume_24h=5000,
            token_ids=["tok-1"],
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await ranging_strategy.generate_signal(context)
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_insufficient_change(self, trending_up_strategy: MomentumStrategy):
        """Small price change should not generate signal."""
        # Change = 0.01 < 0.025 threshold
        history = [0.50] * 12 + [0.51]
        market = MarketData(
            market_id="mkt-small",
            probability=0.51,
            volume_24h=5000,
            token_ids=["tok-1"],
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await trending_up_strategy.generate_signal(context)
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_insufficient_data(self, trending_up_strategy: MomentumStrategy):
        """Less than 12 data points should not generate signal."""
        market = MarketData(
            market_id="mkt-nodata",
            probability=0.55,
            volume_24h=5000,
            token_ids=["tok-1"],
        )
        context = MarketContext(market=market, probability_history=[0.50] * 5)
        signal = await trending_up_strategy.generate_signal(context)
        assert signal is None


class TestAntiResolutionFilter:
    @pytest.mark.asyncio
    async def test_filters_converging_near_resolution(self, trending_up_strategy: MomentumStrategy):
        """Market near resolution converging to 1 should be filtered."""
        end_date = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        history = [0.90] * 12 + [0.95]
        market = MarketData(
            market_id="mkt-resolve",
            probability=0.95,  # Converging to 1
            volume_24h=5000,
            token_ids=["tok-1"],
            end_date=end_date,
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await trending_up_strategy.generate_signal(context)
        assert signal is None

    @pytest.mark.asyncio
    async def test_allows_momentum_far_from_resolution(self, trending_up_strategy: MomentumStrategy):
        """Market > 48h from resolution should allow momentum."""
        end_date = (datetime.now(timezone.utc) + timedelta(hours=96)).isoformat()
        history = [0.50] * 12 + [0.55]
        market = MarketData(
            market_id="mkt-far",
            probability=0.55,
            volume_24h=5000,
            token_ids=["tok-1"],
            end_date=end_date,
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await trending_up_strategy.generate_signal(context)
        assert signal is not None


class TestExitConditions:
    def test_exit_when_momentum_fades(self):
        assert MomentumStrategy.check_exit_conditions(0.005) is True

    def test_no_exit_with_strong_momentum(self):
        assert MomentumStrategy.check_exit_conditions(0.03) is False

    def test_exit_at_threshold(self):
        assert MomentumStrategy.check_exit_conditions(MOMENTUM_EXIT_THRESHOLD - 0.001) is True

    def test_no_exit_at_threshold(self):
        assert MomentumStrategy.check_exit_conditions(MOMENTUM_EXIT_THRESHOLD + 0.001) is False
