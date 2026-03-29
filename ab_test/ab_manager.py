"""A/B testing + multi-wallet manager (Module 26 — optional).

Requires capital > $1,000 for statistical significance.

A/B Testing:
  Pool A vs Pool B (50/50), minimum 14 days or 20 trades per pool.
  Winner: p-value < 0.10 AND Sharpe difference > 8%.
  If no clear difference: keep the more conservative variant.

Multi-Wallet: up to 3 wallets with independent strategies.
  Dashboard shows unified PnL + comparative per wallet.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

MIN_CAPITAL_FOR_AB = 1000.0
MIN_TRADES_PER_POOL = 20
MIN_DAYS_AB = 14
P_VALUE_THRESHOLD = 0.10
SHARPE_DIFF_THRESHOLD = 0.08  # 8%
MAX_WALLETS = 3


@dataclass
class ABPool:
    """One side of an A/B test."""

    pool_id: str  # "A" or "B"
    params: dict[str, float]
    trades: list[dict] = field(default_factory=list)
    pnl_usd: float = 0.0
    created_at: str = ""

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def pnls(self) -> list[float]:
        return [t.get("pnl_usd", 0.0) for t in self.trades]

    @property
    def sharpe_ratio(self) -> float:
        pnls = self.pnls
        if len(pnls) < 2:
            return 0.0
        mean_r = statistics.mean(pnls)
        std_r = statistics.stdev(pnls)
        if std_r == 0:
            return 0.0
        return mean_r / std_r * math.sqrt(365)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get("pnl_usd", 0) > 0)
        return wins / len(self.trades)


@dataclass
class ABTestResult:
    """Result of an A/B test evaluation."""

    winner: str | None  # "A", "B", or None if inconclusive
    p_value: float
    sharpe_a: float
    sharpe_b: float
    sharpe_diff_pct: float
    trades_a: int
    trades_b: int
    is_significant: bool
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "p_value": round(self.p_value, 4),
            "sharpe_a": round(self.sharpe_a, 3),
            "sharpe_b": round(self.sharpe_b, 3),
            "sharpe_diff_pct": round(self.sharpe_diff_pct, 2),
            "trades_a": self.trades_a,
            "trades_b": self.trades_b,
            "is_significant": self.is_significant,
            "recommendation": self.recommendation,
        }


@dataclass
class WalletState:
    """State of an individual wallet in multi-wallet mode."""

    wallet_id: str
    strategy: str
    balance: float = 0.0
    total_pnl: float = 0.0
    trades: list[dict] = field(default_factory=list)
    active: bool = True


class ABTestManager:
    """Manages A/B testing between parameter variants and multi-wallet strategies."""

    def __init__(self) -> None:
        self._pool_a: ABPool | None = None
        self._pool_b: ABPool | None = None
        self._test_active = False
        self._test_start: str = ""
        self._last_result: ABTestResult | None = None
        self._wallets: dict[str, WalletState] = {}

    @property
    def is_test_active(self) -> bool:
        return self._test_active

    @property
    def last_result(self) -> ABTestResult | None:
        return self._last_result

    def start_ab_test(
        self, params_a: dict[str, float], params_b: dict[str, float], capital: float
    ) -> bool:
        """Start an A/B test between two parameter variants.

        Args:
            params_a: Parameters for Pool A.
            params_b: Parameters for Pool B.
            capital: Total available capital.

        Returns:
            True if test started, False if capital insufficient.
        """
        if capital < MIN_CAPITAL_FOR_AB:
            logger.warning(
                "ab_test_insufficient_capital",
                capital=capital,
                required=MIN_CAPITAL_FOR_AB,
            )
            return False

        now = datetime.now(timezone.utc).isoformat()
        self._pool_a = ABPool(pool_id="A", params=params_a, created_at=now)
        self._pool_b = ABPool(pool_id="B", params=params_b, created_at=now)
        self._test_active = True
        self._test_start = now
        logger.info("ab_test_started", params_a=params_a, params_b=params_b)
        return True

    def assign_trade(self, trade: dict) -> str:
        """Assign a trade to Pool A or B (alternating 50/50).

        Returns the pool assignment ("A" or "B").
        """
        if not self._test_active or not self._pool_a or not self._pool_b:
            return "A"

        # Assign to the pool with fewer trades
        if self._pool_a.num_trades <= self._pool_b.num_trades:
            self._pool_a.trades.append(trade)
            self._pool_a.pnl_usd += trade.get("pnl_usd", 0)
            return "A"
        else:
            self._pool_b.trades.append(trade)
            self._pool_b.pnl_usd += trade.get("pnl_usd", 0)
            return "B"

    def evaluate(self) -> ABTestResult | None:
        """Evaluate the A/B test for statistical significance.

        Returns ABTestResult if enough data, None if too early.
        """
        if not self._pool_a or not self._pool_b:
            return None

        # Check minimum trades
        if (
            self._pool_a.num_trades < MIN_TRADES_PER_POOL
            or self._pool_b.num_trades < MIN_TRADES_PER_POOL
        ):
            return None

        sharpe_a = self._pool_a.sharpe_ratio
        sharpe_b = self._pool_b.sharpe_ratio

        # Sharpe difference
        base_sharpe = max(abs(sharpe_a), abs(sharpe_b), 0.01)
        sharpe_diff_pct = abs(sharpe_a - sharpe_b) / base_sharpe * 100

        # Welch's t-test
        p_value = self._welch_t_test(self._pool_a.pnls, self._pool_b.pnls)

        is_significant = p_value < P_VALUE_THRESHOLD and sharpe_diff_pct > SHARPE_DIFF_THRESHOLD * 100

        winner = None
        recommendation = "Inconclusive — keep conservative variant"
        if is_significant:
            if sharpe_a > sharpe_b:
                winner = "A"
                recommendation = f"Pool A wins (Sharpe {sharpe_a:.3f} vs {sharpe_b:.3f})"
            else:
                winner = "B"
                recommendation = f"Pool B wins (Sharpe {sharpe_b:.3f} vs {sharpe_a:.3f})"

        result = ABTestResult(
            winner=winner,
            p_value=p_value,
            sharpe_a=sharpe_a,
            sharpe_b=sharpe_b,
            sharpe_diff_pct=sharpe_diff_pct,
            trades_a=self._pool_a.num_trades,
            trades_b=self._pool_b.num_trades,
            is_significant=is_significant,
            recommendation=recommendation,
        )
        self._last_result = result
        return result

    def stop_test(self) -> ABTestResult | None:
        """Stop the A/B test and return final evaluation."""
        result = self.evaluate()
        self._test_active = False
        logger.info("ab_test_stopped", result=result.to_dict() if result else None)
        return result

    @staticmethod
    def _welch_t_test(sample_a: list[float], sample_b: list[float]) -> float:
        """Compute Welch's t-test p-value between two samples."""
        if len(sample_a) < 2 or len(sample_b) < 2:
            return 1.0

        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
            return float(p_value)
        except ImportError:
            # Fallback: manual Welch's t-test
            n1, n2 = len(sample_a), len(sample_b)
            mean1, mean2 = statistics.mean(sample_a), statistics.mean(sample_b)
            var1 = statistics.variance(sample_a)
            var2 = statistics.variance(sample_b)

            se = math.sqrt(var1 / n1 + var2 / n2)
            if se == 0:
                return 1.0

            t_stat = (mean1 - mean2) / se
            # Approximate p-value using normal distribution for large samples
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))
            return p_value

    # ── Multi-Wallet ──────────────────────────────────────────

    def add_wallet(self, wallet_id: str, strategy: str, balance: float) -> bool:
        """Add a wallet for independent strategy tracking."""
        if len(self._wallets) >= MAX_WALLETS:
            logger.warning("max_wallets_reached", max=MAX_WALLETS)
            return False

        self._wallets[wallet_id] = WalletState(
            wallet_id=wallet_id,
            strategy=strategy,
            balance=balance,
        )
        logger.info("wallet_added", wallet_id=wallet_id, strategy=strategy)
        return True

    def remove_wallet(self, wallet_id: str) -> bool:
        if wallet_id not in self._wallets:
            return False
        del self._wallets[wallet_id]
        return True

    def record_wallet_trade(self, wallet_id: str, trade: dict) -> None:
        """Record a trade for a specific wallet."""
        wallet = self._wallets.get(wallet_id)
        if not wallet:
            return
        wallet.trades.append(trade)
        wallet.total_pnl += trade.get("pnl_usd", 0.0)

    def get_unified_pnl(self) -> dict:
        """Get consolidated PnL across all wallets."""
        total_pnl = sum(w.total_pnl for w in self._wallets.values())
        total_balance = sum(w.balance for w in self._wallets.values())
        return {
            "total_pnl": round(total_pnl, 2),
            "total_balance": round(total_balance, 2),
            "wallets": [
                {
                    "wallet_id": w.wallet_id,
                    "strategy": w.strategy,
                    "balance": round(w.balance, 2),
                    "total_pnl": round(w.total_pnl, 2),
                    "num_trades": len(w.trades),
                    "active": w.active,
                }
                for w in self._wallets.values()
            ],
        }

    def get_status(self) -> dict:
        """Get full A/B test + multi-wallet status."""
        result = {
            "ab_test_active": self._test_active,
            "wallets": len(self._wallets),
        }
        if self._test_active and self._pool_a and self._pool_b:
            result["pool_a"] = {
                "trades": self._pool_a.num_trades,
                "pnl": round(self._pool_a.pnl_usd, 2),
                "sharpe": round(self._pool_a.sharpe_ratio, 3),
                "win_rate": round(self._pool_a.win_rate, 3),
            }
            result["pool_b"] = {
                "trades": self._pool_b.num_trades,
                "pnl": round(self._pool_b.pnl_usd, 2),
                "sharpe": round(self._pool_b.sharpe_ratio, 3),
                "win_rate": round(self._pool_b.win_rate, 3),
            }
        if self._last_result:
            result["last_result"] = self._last_result.to_dict()
        return result
