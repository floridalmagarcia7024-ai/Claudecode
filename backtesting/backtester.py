"""Backtesting engine using DataCollector historical data (Module 15).

Runs strategies against collected market snapshots to validate performance
before deploying in paper or real mode.

Activation thresholds:
  Sharpe >= 0.4
  MaxDD <= 35%
  Trades >= 20
  p-value <= 0.10
  Profit factor >= 1.1
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import structlog
from scipy import stats as scipy_stats

from config import settings

logger = structlog.get_logger(__name__)

SHARPE_MIN = 0.4
MAX_DD_PCT = 35.0
MIN_TRADES = 20
P_VALUE_MAX = 0.10
PROFIT_FACTOR_MIN = 1.1
RISK_FREE_RATE = 0.04  # ~4% annualized


@dataclass
class BacktestTrade:
    """Single trade in a backtest."""

    market_id: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size_usd: float = 100.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees_usd: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    strategy: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Full backtest result with metrics."""

    strategy: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    p_value: float = 1.0
    passes_thresholds: bool = False
    semaphore: Literal["GREEN", "YELLOW", "RED"] = "RED"
    failure_reasons: list[str] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "win_rate": round(self.win_rate, 3),
            "avg_win": round(self.avg_win, 3),
            "avg_loss": round(self.avg_loss, 3),
            "p_value": round(self.p_value, 4),
            "passes_thresholds": self.passes_thresholds,
            "semaphore": self.semaphore,
            "failure_reasons": self.failure_reasons,
        }


class Backtester:
    """Backtesting engine that replays DataCollector history through strategies.

    Uses market snapshot data from SQLite (collected by DataCollector).
    """

    def __init__(self) -> None:
        self._results: dict[str, BacktestResult] = {}

    def get_result(self, strategy: str) -> BacktestResult | None:
        return self._results.get(strategy)

    async def run_backtest(
        self,
        strategy_name: str,
        snapshots: list[dict],
        params: dict | None = None,
    ) -> BacktestResult:
        """Run a backtest for a strategy on historical snapshots.

        Args:
            strategy_name: "mean_reversion" or "momentum".
            snapshots: List of dicts with timestamp, probability, volume_24h, spread_pct.
            params: Override parameters for the strategy.

        Returns:
            BacktestResult with full metrics.
        """
        params = params or {}
        if strategy_name == "mean_reversion":
            trades = self._backtest_mean_reversion(snapshots, params)
        elif strategy_name == "momentum":
            trades = self._backtest_momentum(snapshots, params)
        else:
            logger.warning("unknown_strategy_backtest", strategy=strategy_name)
            return BacktestResult(strategy=strategy_name)

        result = self._calculate_metrics(strategy_name, trades)
        self._results[strategy_name] = result

        logger.info(
            "backtest_complete",
            strategy=strategy_name,
            trades=result.total_trades,
            sharpe=round(result.sharpe_ratio, 3),
            max_dd=round(result.max_drawdown_pct, 2),
            semaphore=result.semaphore,
        )
        return result

    def _backtest_mean_reversion(
        self, snapshots: list[dict], params: dict
    ) -> list[BacktestTrade]:
        """Simulate mean reversion strategy on historical data."""
        zscore_threshold = params.get("zscore_threshold", settings.zscore_threshold)
        exit_zscore = params.get("exit_zscore", 0.5)
        lookback = params.get("lookback", 20)
        fee_pct = settings.paper_fee_pct / 100

        trades: list[BacktestTrade] = []
        position: BacktestTrade | None = None

        probabilities = [s["probability"] for s in snapshots]

        for i in range(lookback, len(probabilities)):
            window = probabilities[i - lookback : i]
            current = probabilities[i]
            ts = snapshots[i].get("timestamp", "")

            mean = statistics.mean(window)
            std = statistics.stdev(window)
            if std < 0.001:
                continue

            z_score = (current - mean) / std

            if position is None:
                # Entry signal
                if z_score < -zscore_threshold:
                    position = BacktestTrade(
                        direction="BUY",
                        entry_price=current,
                        entry_time=ts,
                        strategy="mean_reversion",
                        size_usd=100.0,
                    )
                elif z_score > zscore_threshold:
                    position = BacktestTrade(
                        direction="SELL",
                        entry_price=current,
                        entry_time=ts,
                        strategy="mean_reversion",
                        size_usd=100.0,
                    )
            else:
                # Exit signal: z-score reverts
                should_exit = -exit_zscore <= z_score <= exit_zscore
                if should_exit:
                    if position.direction == "BUY":
                        pnl_pct = (current - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - current) / position.entry_price

                    fees = position.size_usd * fee_pct * 2  # entry + exit
                    pnl_usd = position.size_usd * pnl_pct - fees

                    position.exit_price = current
                    position.exit_time = ts
                    position.pnl_usd = pnl_usd
                    position.pnl_pct = pnl_pct * 100
                    position.fees_usd = fees
                    position.exit_reason = "mean_reversion_exit"
                    trades.append(position)
                    position = None

        return trades

    def _backtest_momentum(
        self, snapshots: list[dict], params: dict
    ) -> list[BacktestTrade]:
        """Simulate momentum strategy on historical data."""
        change_threshold = params.get("change_threshold", 0.025)
        exit_threshold = params.get("exit_threshold", 0.01)
        lookback_points = params.get("lookback_points", 12)  # 1h at 5-min
        fee_pct = settings.paper_fee_pct / 100

        trades: list[BacktestTrade] = []
        position: BacktestTrade | None = None

        probabilities = [s["probability"] for s in snapshots]

        for i in range(lookback_points, len(probabilities)):
            current = probabilities[i]
            past = probabilities[i - lookback_points]
            change = current - past
            ts = snapshots[i].get("timestamp", "")

            if position is None:
                if abs(change) >= change_threshold:
                    direction = "BUY" if change > 0 else "SELL"
                    position = BacktestTrade(
                        direction=direction,
                        entry_price=current,
                        entry_time=ts,
                        strategy="momentum",
                        size_usd=100.0,
                    )
            else:
                # Exit when momentum fades
                if abs(change) < exit_threshold:
                    if position.direction == "BUY":
                        pnl_pct = (current - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - current) / position.entry_price

                    fees = position.size_usd * fee_pct * 2
                    pnl_usd = position.size_usd * pnl_pct - fees

                    position.exit_price = current
                    position.exit_time = ts
                    position.pnl_usd = pnl_usd
                    position.pnl_pct = pnl_pct * 100
                    position.fees_usd = fees
                    position.exit_reason = "momentum_exit"
                    trades.append(position)
                    position = None

        return trades

    def _calculate_metrics(
        self, strategy: str, trades: list[BacktestTrade]
    ) -> BacktestResult:
        """Calculate all performance metrics from trade list."""
        result = BacktestResult(strategy=strategy, trades=trades)
        result.total_trades = len(trades)

        if not trades:
            result.failure_reasons.append(f"No trades (need >= {MIN_TRADES})")
            result.semaphore = "RED"
            return result

        pnls = [t.pnl_usd for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.total_pnl = sum(pnls)
        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_win = statistics.mean(wins) if wins else 0
        result.avg_loss = statistics.mean([abs(l) for l in losses]) if losses else 0

        # Equity curve
        equity = [0.0]
        for pnl in pnls:
            equity.append(equity[-1] + pnl)
        result.equity_curve = equity

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for val in equity:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
        initial_capital = 10000.0  # Notional for DD%
        result.max_drawdown_pct = (max_dd / initial_capital) * 100 if initial_capital > 0 else 0

        # Sharpe ratio (annualized, assuming ~365 trading days)
        if len(pnls) >= 2:
            daily_returns = [p / 100 for p in pnls]  # Normalized
            mean_ret = statistics.mean(daily_returns)
            std_ret = statistics.stdev(daily_returns)
            if std_ret > 0:
                result.sharpe_ratio = (mean_ret - RISK_FREE_RATE / 365) / std_ret * math.sqrt(365)
            else:
                result.sharpe_ratio = 0.0
        else:
            result.sharpe_ratio = 0.0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = sum(abs(l) for l in losses)
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # P-value (t-test: are returns significantly different from 0?)
        if len(pnls) >= 2:
            t_stat, p_value = scipy_stats.ttest_1samp(pnls, 0)
            result.p_value = p_value
        else:
            result.p_value = 1.0

        # Evaluate thresholds
        failures = []
        if result.total_trades < MIN_TRADES:
            failures.append(f"Trades {result.total_trades} < {MIN_TRADES}")
        if result.sharpe_ratio < SHARPE_MIN:
            failures.append(f"Sharpe {result.sharpe_ratio:.3f} < {SHARPE_MIN}")
        if result.max_drawdown_pct > MAX_DD_PCT:
            failures.append(f"MaxDD {result.max_drawdown_pct:.1f}% > {MAX_DD_PCT}%")
        if result.p_value > P_VALUE_MAX:
            failures.append(f"p-value {result.p_value:.4f} > {P_VALUE_MAX}")
        if result.profit_factor < PROFIT_FACTOR_MIN:
            failures.append(f"Profit factor {result.profit_factor:.3f} < {PROFIT_FACTOR_MIN}")

        result.failure_reasons = failures
        result.passes_thresholds = len(failures) == 0

        # Semaphore
        if result.passes_thresholds:
            result.semaphore = "GREEN"
        elif len(failures) <= 2:
            result.semaphore = "YELLOW"
        else:
            result.semaphore = "RED"

        return result
