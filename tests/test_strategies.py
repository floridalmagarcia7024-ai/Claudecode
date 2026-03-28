"""Tests for mean reversion strategy."""

from __future__ import annotations

import random
import statistics

import pytest
import pytest_asyncio

from api.polymarket import MarketData, OrderBook
from core.state import PortfolioState
from strategies.base import MarketContext
from strategies.mean_reversion import MeanReversionStrategy, MIN_DATA_POINTS


@pytest.fixture
def strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(ai_analyzer=None)


class TestZScoreComputation:
    def test_z_score_normal(self, strategy: MeanReversionStrategy):
        history = [0.60] * 200
        # Current at mean → z-score should be None (zero variance)
        result = MeanReversionStrategy.compute_z_score(0.60, history)
        assert result is None  # zero variance

    def test_z_score_with_variance(self):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        current = mean + 2 * std  # ~2 standard deviations above
        z = MeanReversionStrategy.compute_z_score(current, history)
        assert z is not None
        assert z > 1.5

    def test_z_score_insufficient_data(self):
        history = [0.50] * 5
        assert MeanReversionStrategy.compute_z_score(0.50, history) is None

    def test_z_score_negative(self):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        current = mean - 2.5 * std  # Below mean
        z = MeanReversionStrategy.compute_z_score(current, history)
        assert z is not None
        assert z < -1.5


class TestExitConditions:
    def test_exit_at_zero(self):
        assert MeanReversionStrategy.check_exit_conditions(0.0) is True

    def test_exit_at_boundary(self):
        assert MeanReversionStrategy.check_exit_conditions(0.5) is True
        assert MeanReversionStrategy.check_exit_conditions(-0.5) is True

    def test_no_exit_outside_range(self):
        assert MeanReversionStrategy.check_exit_conditions(1.0) is False
        assert MeanReversionStrategy.check_exit_conditions(-1.0) is False


class TestSignalGeneration:
    @pytest.mark.asyncio
    async def test_no_signal_insufficient_data(self, strategy: MeanReversionStrategy):
        market = MarketData(
            market_id="test", probability=0.65, volume_24h=5000, token_ids=["t1"]
        )
        context = MarketContext(
            market=market, probability_history=[0.60] * 5  # Too few
        )
        signal = await strategy.generate_signal(context)
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_below_threshold(self, strategy: MeanReversionStrategy):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        # Current close to mean → z-score < threshold
        market = MarketData(
            market_id="test", probability=mean + 0.01, volume_24h=5000, token_ids=["t1"]
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await strategy.generate_signal(context)
        assert signal is None

    @pytest.mark.asyncio
    async def test_buy_signal_on_low_z_score(self, strategy: MeanReversionStrategy):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        # Current well below mean
        low_price = mean - 2.5 * std
        market = MarketData(
            market_id="test",
            condition_id="cond",
            probability=low_price,
            volume_24h=5000,
            token_ids=["t1"],
            category="crypto",
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await strategy.generate_signal(context)
        assert signal is not None
        assert signal.direction == "BUY"
        assert signal.z_score is not None
        assert signal.z_score < 0

    @pytest.mark.asyncio
    async def test_sell_signal_on_high_z_score(self, strategy: MeanReversionStrategy):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        high_price = mean + 2.5 * std
        market = MarketData(
            market_id="test",
            condition_id="cond",
            probability=high_price,
            volume_24h=5000,
            token_ids=["t1"],
            category="crypto",
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await strategy.generate_signal(context)
        assert signal is not None
        assert signal.direction == "SELL"

    @pytest.mark.asyncio
    async def test_reject_low_volume(self, strategy: MeanReversionStrategy):
        random.seed(42)
        history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
        mean = statistics.mean(history)
        std = statistics.stdev(history)
        market = MarketData(
            market_id="test",
            probability=mean - 2.5 * std,
            volume_24h=100,  # Below MIN_DAILY_VOLUME
            token_ids=["t1"],
        )
        context = MarketContext(market=market, probability_history=history)
        signal = await strategy.generate_signal(context)
        assert signal is None


class TestPositionSizing:
    def test_default_sizing(self, strategy: MeanReversionStrategy):
        from strategies.base import Signal
        signal = Signal(market_id="test", direction="BUY", strategy="mean_reversion")
        portfolio = PortfolioState(capital=10_000)
        size = strategy.calculate_size(signal, portfolio)
        assert size == pytest.approx(500.0, abs=1)  # 5% of 10000
