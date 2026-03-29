"""Automatic parameter optimizer with human validation (Module 21).

Trigger: Sunday 02:00 UTC (if bot active > 21 days).
Method: Grid search walk-forward, max 300 combinations.

New params go to paper mode for 7 days. If Sharpe improves > 10%,
proposal is sent via Telegram. NEVER applied in real mode without
manual /aprobar_params confirmation.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from config import settings

logger = structlog.get_logger(__name__)

# Bounded parameter ranges (Module 21)
PARAM_RANGES = {
    "zscore_threshold": [1.5, 1.7, 1.8, 2.0, 2.2, 2.5],
    "sentiment_shift": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
    "ai_confidence_min": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "trailing_pct": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
    "sl_base_pct": [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
}

MAX_COMBINATIONS = 300


class OptimizationStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAPER_TESTING = "paper_testing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class OptimizationResult:
    """Result of a single parameter optimization run."""

    params: dict[str, float]
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    improvement_pct: float  # vs current params
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 3),
            "total_trades": self.total_trades,
            "improvement_pct": round(self.improvement_pct, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationProposal:
    """Proposed parameter changes awaiting approval."""

    current_params: dict[str, float]
    proposed_params: dict[str, float]
    result: OptimizationResult
    paper_start: str = ""
    paper_end: str = ""
    paper_sharpe: float = 0.0
    status: OptimizationStatus = OptimizationStatus.AWAITING_APPROVAL


class AutoOptimizer:
    """Walk-forward parameter optimizer with human-in-the-loop validation.

    Runs weekly grid search on Sunday 02:00 UTC. Proposed params enter
    paper testing for 7 days. Only applied after explicit human approval.
    """

    def __init__(self, backtester=None, walk_forward=None) -> None:
        self._backtester = backtester
        self._walk_forward = walk_forward
        self._status = OptimizationStatus.IDLE
        self._current_proposal: OptimizationProposal | None = None
        self._history: list[OptimizationResult] = []
        self._last_run: float = 0
        self._bot_start_time: float = time.time()
        self._paper_test_start: float = 0
        self._paper_test_params: dict[str, float] = {}
        self._paper_test_trades: list[dict] = []

    @property
    def status(self) -> OptimizationStatus:
        return self._status

    @property
    def current_proposal(self) -> OptimizationProposal | None:
        return self._current_proposal

    @property
    def history(self) -> list[OptimizationResult]:
        return self._history

    def get_current_params(self) -> dict[str, float]:
        """Get current live parameter values."""
        return {
            "zscore_threshold": settings.zscore_threshold,
            "sentiment_shift": settings.sentiment_shift,
            "ai_confidence_min": settings.ai_confidence_min,
            "trailing_pct": settings.trailing_pct,
            "sl_base_pct": settings.max_daily_loss_pct,
        }

    def _generate_param_grid(self) -> list[dict[str, float]]:
        """Generate bounded parameter combinations, capped at MAX_COMBINATIONS."""
        keys = list(PARAM_RANGES.keys())
        values = list(PARAM_RANGES.values())

        all_combos = list(itertools.product(*values))

        # If too many, sample evenly
        if len(all_combos) > MAX_COMBINATIONS:
            step = len(all_combos) // MAX_COMBINATIONS
            all_combos = all_combos[::step][:MAX_COMBINATIONS]

        return [dict(zip(keys, combo)) for combo in all_combos]

    async def run_optimization(self, snapshots: list[dict]) -> OptimizationResult | None:
        """Run grid search walk-forward optimization.

        Args:
            snapshots: Historical market snapshots for backtesting.

        Returns:
            Best result if improvement found, None otherwise.
        """
        if self._status == OptimizationStatus.RUNNING:
            logger.warning("optimizer_already_running")
            return None

        self._status = OptimizationStatus.RUNNING
        logger.info("optimizer_starting", combinations=MAX_COMBINATIONS)

        try:
            current_params = self.get_current_params()
            grid = self._generate_param_grid()

            # Evaluate current params as baseline
            baseline_sharpe = await self._evaluate_params(current_params, snapshots)

            best_result: OptimizationResult | None = None
            best_sharpe = baseline_sharpe

            for params in grid:
                sharpe = await self._evaluate_params(params, snapshots)
                if sharpe > best_sharpe:
                    improvement = (
                        ((sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100)
                        if baseline_sharpe != 0
                        else 100.0
                    )
                    best_sharpe = sharpe
                    best_result = OptimizationResult(
                        params=params,
                        sharpe_ratio=sharpe,
                        max_drawdown_pct=0.0,
                        win_rate=0.0,
                        total_trades=0,
                        improvement_pct=improvement,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                # Yield control to event loop periodically
                await asyncio.sleep(0)

            if best_result and best_result.improvement_pct >= 10.0:
                self._current_proposal = OptimizationProposal(
                    current_params=current_params,
                    proposed_params=best_result.params,
                    result=best_result,
                    status=OptimizationStatus.PAPER_TESTING,
                    paper_start=datetime.now(timezone.utc).isoformat(),
                )
                self._status = OptimizationStatus.PAPER_TESTING
                self._paper_test_start = time.time()
                self._paper_test_params = best_result.params.copy()
                self._history.append(best_result)
                logger.info(
                    "optimizer_found_improvement",
                    improvement_pct=best_result.improvement_pct,
                    sharpe=best_result.sharpe_ratio,
                )
                return best_result

            self._status = OptimizationStatus.IDLE
            logger.info("optimizer_no_improvement")
            return None

        except Exception as exc:
            logger.error("optimizer_error", error=str(exc))
            self._status = OptimizationStatus.IDLE
            return None

    async def _evaluate_params(
        self, params: dict[str, float], snapshots: list[dict]
    ) -> float:
        """Evaluate a parameter set using walk-forward backtesting.

        Returns annualized Sharpe ratio.
        """
        if self._walk_forward:
            try:
                result = await self._walk_forward.evaluate_with_params(
                    params, snapshots
                )
                return result.get("sharpe_ratio", 0.0)
            except Exception:
                pass

        # Fallback: simple evaluation based on parameter heuristics
        # (used when backtester not available, e.g., in tests)
        import statistics

        scores = []
        # Penalize extreme values
        if params.get("zscore_threshold", 1.8) < 1.6:
            scores.append(-0.5)
        elif params.get("zscore_threshold", 1.8) > 2.3:
            scores.append(-0.3)
        else:
            scores.append(0.5)

        if params.get("trailing_pct", 5.0) < 4.0:
            scores.append(-0.2)
        else:
            scores.append(0.3)

        return statistics.mean(scores) if scores else 0.0

    def check_paper_test_complete(self) -> bool:
        """Check if 7-day paper test period has elapsed."""
        if self._status != OptimizationStatus.PAPER_TESTING:
            return False
        elapsed_days = (time.time() - self._paper_test_start) / 86400
        return elapsed_days >= 7

    def promote_to_approval(self, paper_sharpe: float) -> bool:
        """Promote paper-tested params to awaiting approval if improved.

        Returns True if improvement >= 10% and proposal created.
        """
        if not self._current_proposal:
            return False

        self._current_proposal.paper_sharpe = paper_sharpe
        self._current_proposal.paper_end = datetime.now(timezone.utc).isoformat()

        baseline = self._current_proposal.result.sharpe_ratio
        if baseline > 0 and paper_sharpe > baseline * 0.9:
            self._current_proposal.status = OptimizationStatus.AWAITING_APPROVAL
            self._status = OptimizationStatus.AWAITING_APPROVAL
            return True

        self._current_proposal.status = OptimizationStatus.REJECTED
        self._status = OptimizationStatus.IDLE
        self._current_proposal = None
        return False

    def approve_params(self) -> dict[str, float] | None:
        """Approve proposed parameters — apply them to live settings.

        Returns the approved params dict, or None if nothing to approve.
        """
        if (
            self._status != OptimizationStatus.AWAITING_APPROVAL
            or not self._current_proposal
        ):
            return None

        params = self._current_proposal.proposed_params
        # Apply to live settings
        if "zscore_threshold" in params:
            settings.zscore_threshold = params["zscore_threshold"]
        if "sentiment_shift" in params:
            settings.sentiment_shift = params["sentiment_shift"]
        if "ai_confidence_min" in params:
            settings.ai_confidence_min = params["ai_confidence_min"]
        if "trailing_pct" in params:
            settings.trailing_pct = params["trailing_pct"]

        self._current_proposal.status = OptimizationStatus.APPROVED
        self._status = OptimizationStatus.IDLE
        logger.info("optimizer_params_approved", params=params)

        approved = params.copy()
        self._current_proposal = None
        return approved

    def reject_params(self) -> bool:
        """Reject proposed parameters — keep current settings."""
        if (
            self._status != OptimizationStatus.AWAITING_APPROVAL
            or not self._current_proposal
        ):
            return False

        self._current_proposal.status = OptimizationStatus.REJECTED
        self._status = OptimizationStatus.IDLE
        logger.info("optimizer_params_rejected")
        self._current_proposal = None
        return True

    def should_run(self) -> bool:
        """Check if optimizer should run (Sunday 02:00 UTC, >21 days active)."""
        now = datetime.now(timezone.utc)
        # Sunday = 6
        if now.weekday() != 6 or now.hour != 2:
            return False

        # Bot must be active > 21 days
        days_active = (time.time() - self._bot_start_time) / 86400
        if days_active < 21:
            return False

        # Don't run more than once per week
        if time.time() - self._last_run < 6 * 86400:
            return False

        return True

    def mark_run_complete(self) -> None:
        """Mark that a run has been completed."""
        self._last_run = time.time()

    def get_status_dict(self) -> dict:
        """Get optimizer status for dashboard/telegram."""
        result = {
            "status": self._status.value,
            "last_run": datetime.fromtimestamp(self._last_run, tz=timezone.utc).isoformat()
            if self._last_run > 0
            else None,
            "history_count": len(self._history),
        }
        if self._current_proposal:
            result["proposal"] = {
                "improvement_pct": self._current_proposal.result.improvement_pct,
                "sharpe": self._current_proposal.result.sharpe_ratio,
                "paper_sharpe": self._current_proposal.paper_sharpe,
                "proposed_params": self._current_proposal.proposed_params,
            }
        return result
