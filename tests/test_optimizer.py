"""Tests for AutoOptimizer (Module 21)."""

import asyncio
import pytest

from optimizer.auto_optimizer import (
    AutoOptimizer,
    OptimizationStatus,
    PARAM_RANGES,
    MAX_COMBINATIONS,
)


class TestAutoOptimizer:
    def setup_method(self):
        self.optimizer = AutoOptimizer()

    def test_initial_status_idle(self):
        assert self.optimizer.status == OptimizationStatus.IDLE
        assert self.optimizer.current_proposal is None

    def test_get_current_params(self):
        params = self.optimizer.get_current_params()
        assert "zscore_threshold" in params
        assert "sentiment_shift" in params
        assert "ai_confidence_min" in params
        assert "trailing_pct" in params

    def test_generate_param_grid_capped(self):
        grid = self.optimizer._generate_param_grid()
        assert len(grid) <= MAX_COMBINATIONS
        assert len(grid) > 0
        # Each combo has all param keys
        for combo in grid[:5]:
            for key in PARAM_RANGES:
                assert key in combo

    def test_should_run_not_enough_days(self):
        # Bot just started — should not run
        assert not self.optimizer.should_run()

    def test_approve_params_nothing_pending(self):
        result = self.optimizer.approve_params()
        assert result is None

    def test_reject_params_nothing_pending(self):
        result = self.optimizer.reject_params()
        assert result is False

    @pytest.mark.asyncio
    async def test_run_optimization_empty_snapshots(self):
        result = await self.optimizer.run_optimization([])
        # With empty snapshots and fallback evaluator, may or may not find improvement
        assert self.optimizer.status in (
            OptimizationStatus.IDLE,
            OptimizationStatus.PAPER_TESTING,
        )

    @pytest.mark.asyncio
    async def test_run_optimization_twice_rejects(self):
        # Start a run
        self.optimizer._status = OptimizationStatus.RUNNING
        result = await self.optimizer.run_optimization([])
        assert result is None  # Already running

    def test_check_paper_test_not_testing(self):
        assert not self.optimizer.check_paper_test_complete()

    def test_promote_to_approval_no_proposal(self):
        assert not self.optimizer.promote_to_approval(1.5)

    def test_get_status_dict(self):
        status = self.optimizer.get_status_dict()
        assert status["status"] == "idle"
        assert status["history_count"] == 0

    def test_mark_run_complete(self):
        self.optimizer.mark_run_complete()
        assert self.optimizer._last_run > 0
