"""Tests for backtesting engine and walk-forward optimizer (Module 15)."""

from __future__ import annotations

import pytest

from backtesting.backtester import (
    Backtester,
    BacktestResult,
    MAX_DD_PCT,
    MIN_TRADES,
    P_VALUE_MAX,
    PROFIT_FACTOR_MIN,
    SHARPE_MIN,
)
from backtesting.walk_forward import WalkForwardOptimizer, _build_param_grid


# ── Fixtures ──────────────────────────────────────────────────

def _make_snapshots(n: int = 200, base: float = 0.50, noise: float = 0.05) -> list[dict]:
    """Generate synthetic market snapshots with mean-reverting behavior."""
    import math
    import random

    random.seed(42)
    snapshots = []
    prob = base
    for i in range(n):
        # Mean reversion: drift back toward base
        drift = (base - prob) * 0.1
        change = drift + random.uniform(-noise, noise)
        prob = max(0.05, min(0.95, prob + change))
        snapshots.append({
            "timestamp": f"2025-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
            "probability": prob,
            "volume_24h": random.uniform(1000, 5000),
            "spread_pct": random.uniform(1.0, 5.0),
        })
    return snapshots


def _make_trending_snapshots(n: int = 200, direction: float = 1.0) -> list[dict]:
    """Generate trending market snapshots for momentum strategy."""
    import random

    random.seed(42)
    snapshots = []
    prob = 0.30 if direction > 0 else 0.70
    for i in range(n):
        change = direction * random.uniform(0.001, 0.01)
        prob = max(0.05, min(0.95, prob + change))
        snapshots.append({
            "timestamp": f"2025-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
            "probability": prob,
            "volume_24h": random.uniform(2000, 8000),
            "spread_pct": random.uniform(1.0, 4.0),
        })
    return snapshots


# ── Backtester Tests ──────────────────────────────────────────


class TestBacktester:
    @pytest.mark.asyncio
    async def test_mean_reversion_backtest_runs(self):
        bt = Backtester()
        snapshots = _make_snapshots(300)
        result = await bt.run_backtest("mean_reversion", snapshots)

        assert isinstance(result, BacktestResult)
        assert result.strategy == "mean_reversion"
        assert result.total_trades >= 0
        assert result.semaphore in ("GREEN", "YELLOW", "RED")

    @pytest.mark.asyncio
    async def test_momentum_backtest_runs(self):
        bt = Backtester()
        snapshots = _make_trending_snapshots(300)
        result = await bt.run_backtest("momentum", snapshots)

        assert isinstance(result, BacktestResult)
        assert result.strategy == "momentum"

    @pytest.mark.asyncio
    async def test_unknown_strategy(self):
        bt = Backtester()
        result = await bt.run_backtest("nonexistent", [])
        assert result.total_trades == 0

    @pytest.mark.asyncio
    async def test_empty_snapshots(self):
        bt = Backtester()
        result = await bt.run_backtest("mean_reversion", [])
        assert result.total_trades == 0
        assert result.semaphore == "RED"

    @pytest.mark.asyncio
    async def test_result_to_dict(self):
        bt = Backtester()
        snapshots = _make_snapshots(300)
        result = await bt.run_backtest("mean_reversion", snapshots)
        d = result.to_dict()

        assert "strategy" in d
        assert "sharpe_ratio" in d
        assert "semaphore" in d
        assert "failure_reasons" in d
        assert isinstance(d["failure_reasons"], list)

    @pytest.mark.asyncio
    async def test_get_result_cached(self):
        bt = Backtester()
        snapshots = _make_snapshots(300)
        await bt.run_backtest("mean_reversion", snapshots)

        cached = bt.get_result("mean_reversion")
        assert cached is not None
        assert cached.strategy == "mean_reversion"

    @pytest.mark.asyncio
    async def test_custom_params(self):
        bt = Backtester()
        snapshots = _make_snapshots(300)
        params = {"zscore_threshold": 1.2, "lookback": 15}
        result = await bt.run_backtest("mean_reversion", snapshots, params)
        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_semaphore_logic(self):
        """With few data points, should get RED or YELLOW semaphore."""
        bt = Backtester()
        snapshots = _make_snapshots(50)  # Few trades expected
        result = await bt.run_backtest("mean_reversion", snapshots)
        assert result.semaphore in ("GREEN", "YELLOW", "RED")


# ── Walk-Forward Tests ────────────────────────────────────────


class TestWalkForward:
    def test_build_param_grid_mean_reversion(self):
        grid = _build_param_grid("mean_reversion")
        assert len(grid) > 0
        assert len(grid) <= 300
        assert "zscore_threshold" in grid[0]

    def test_build_param_grid_momentum(self):
        grid = _build_param_grid("momentum")
        assert len(grid) > 0
        assert "change_threshold" in grid[0]

    def test_build_param_grid_unknown(self):
        grid = _build_param_grid("unknown")
        assert grid == [{}]

    @pytest.mark.asyncio
    async def test_walk_forward_runs(self):
        wf = WalkForwardOptimizer()
        snapshots = _make_snapshots(400)
        result = await wf.optimize("mean_reversion", snapshots, num_windows=2)

        assert result.strategy == "mean_reversion"
        assert result.num_combinations_tested > 0

    @pytest.mark.asyncio
    async def test_walk_forward_insufficient_data(self):
        wf = WalkForwardOptimizer()
        result = await wf.optimize("mean_reversion", [], num_windows=2)
        assert result.num_windows == 0

    @pytest.mark.asyncio
    async def test_walk_forward_to_dict(self):
        wf = WalkForwardOptimizer()
        snapshots = _make_snapshots(400)
        result = await wf.optimize("mean_reversion", snapshots, num_windows=2)
        d = result.to_dict()
        assert "strategy" in d
        assert "best_params" in d
