"""Walk-forward optimization to prevent overfitting (Module 15).

Splits data into sliding windows (75% train + 25% validation).
Grid search over parameter combinations (max 300).
Score = validation_sharpe if mean(train_sharpes) > 0.3, else -inf.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import structlog

from backtesting.backtester import Backtester, BacktestResult

logger = structlog.get_logger(__name__)

MAX_COMBINATIONS = 300
TRAIN_PCT = 0.75
MIN_TRAIN_SHARPE = 0.3


@dataclass
class WalkForwardResult:
    """Result of walk-forward optimization."""

    strategy: str = ""
    best_params: dict = field(default_factory=dict)
    best_validation_sharpe: float = -float("inf")
    num_windows: int = 0
    num_combinations_tested: int = 0
    window_results: list[dict] = field(default_factory=list)
    final_backtest: BacktestResult | None = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "best_params": self.best_params,
            "best_validation_sharpe": round(self.best_validation_sharpe, 4),
            "num_windows": self.num_windows,
            "num_combinations_tested": self.num_combinations_tested,
            "window_results": self.window_results,
            "final_backtest": self.final_backtest.to_dict() if self.final_backtest else None,
        }


def _build_param_grid(strategy: str) -> list[dict]:
    """Build parameter grid for a strategy (max 300 combinations)."""
    if strategy == "mean_reversion":
        grid = {
            "zscore_threshold": [1.2, 1.5, 1.8, 2.0, 2.5],
            "exit_zscore": [0.3, 0.5, 0.7],
            "lookback": [15, 20, 30, 50],
        }
    elif strategy == "momentum":
        grid = {
            "change_threshold": [0.015, 0.020, 0.025, 0.030, 0.040],
            "exit_threshold": [0.005, 0.010, 0.015],
            "lookback_points": [6, 12, 18, 24],
        }
    else:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    # Cap at MAX_COMBINATIONS
    if len(combos) > MAX_COMBINATIONS:
        step = len(combos) // MAX_COMBINATIONS
        combos = combos[::step][:MAX_COMBINATIONS]

    return [dict(zip(keys, combo)) for combo in combos]


class WalkForwardOptimizer:
    """Walk-forward optimization with sliding windows."""

    def __init__(self) -> None:
        self._backtester = Backtester()

    async def optimize(
        self,
        strategy: str,
        snapshots: list[dict],
        num_windows: int = 3,
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        Args:
            strategy: Strategy name.
            snapshots: Full historical snapshots.
            num_windows: Number of sliding windows.

        Returns:
            WalkForwardResult with best parameters.
        """
        result = WalkForwardResult(strategy=strategy)

        if len(snapshots) < 100:
            logger.warning("walk_forward_insufficient_data", count=len(snapshots))
            return result

        param_grid = _build_param_grid(strategy)
        result.num_combinations_tested = len(param_grid)

        # Split into windows
        window_size = len(snapshots) // num_windows
        if window_size < 50:
            num_windows = max(1, len(snapshots) // 50)
            window_size = len(snapshots) // num_windows

        result.num_windows = num_windows

        best_score = -float("inf")
        best_params: dict = {}

        for params in param_grid:
            train_sharpes: list[float] = []
            val_sharpes: list[float] = []

            for w in range(num_windows):
                start = w * window_size
                end = min(start + window_size, len(snapshots))
                window_data = snapshots[start:end]

                split_idx = int(len(window_data) * TRAIN_PCT)
                train_data = window_data[:split_idx]
                val_data = window_data[split_idx:]

                if len(train_data) < 30 or len(val_data) < 10:
                    continue

                # Train
                train_result = await self._backtester.run_backtest(
                    strategy, train_data, params
                )
                train_sharpes.append(train_result.sharpe_ratio)

                # Validate
                val_result = await self._backtester.run_backtest(
                    strategy, val_data, params
                )
                val_sharpes.append(val_result.sharpe_ratio)

            if not train_sharpes:
                continue

            mean_train_sharpe = sum(train_sharpes) / len(train_sharpes)

            # Score: validation sharpe if train is good enough
            if mean_train_sharpe > MIN_TRAIN_SHARPE and val_sharpes:
                score = sum(val_sharpes) / len(val_sharpes)
            else:
                score = -float("inf")

            if score > best_score:
                best_score = score
                best_params = params

        result.best_params = best_params
        result.best_validation_sharpe = best_score

        # Run final backtest with best params on full data
        if best_params:
            result.final_backtest = await self._backtester.run_backtest(
                strategy, snapshots, best_params
            )

        logger.info(
            "walk_forward_complete",
            strategy=strategy,
            best_params=best_params,
            best_val_sharpe=round(best_score, 4),
            windows=num_windows,
            combos_tested=len(param_grid),
        )
        return result
