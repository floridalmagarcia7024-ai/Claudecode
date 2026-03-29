"""Stress testing module — 5 daily scenarios (Module 23).

Runs daily at 00:00 UTC as a non-blocking asyncio task.

Scenarios:
  1. Flash crash: all positions -15% simultaneously
  2. Correlation collapse: correlated positions move opposite
  3. Dry liquidity: spreads at historical max
  4. Black swan: 1 position loses 100%
  5. Dead API (4h): simulate inability to manage positions

For each scenario: simulated PnL, circuit breaker trigger check, max damage.
If loss > 15% of capital → immediate Telegram alert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

LOSS_ALERT_THRESHOLD_PCT = 15.0


@dataclass
class ScenarioResult:
    """Result of a single stress test scenario."""

    scenario_name: str
    description: str
    simulated_pnl: float
    pnl_pct: float  # As % of capital
    triggers_circuit_breaker: bool
    max_damage_usd: float
    positions_affected: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "simulated_pnl": round(self.simulated_pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "triggers_circuit_breaker": self.triggers_circuit_breaker,
            "max_damage_usd": round(self.max_damage_usd, 2),
            "positions_affected": self.positions_affected,
            "details": self.details,
        }


@dataclass
class StressTestReport:
    """Complete daily stress test report."""

    timestamp: str
    capital: float
    scenarios: list[ScenarioResult]
    worst_case_pnl: float
    worst_case_pct: float
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "capital": round(self.capital, 2),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "worst_case_pnl": round(self.worst_case_pnl, 2),
            "worst_case_pct": round(self.worst_case_pct, 2),
            "alerts": self.alerts,
        }


class StressTester:
    """Daily stress testing engine with 5 scenarios.

    Runs as a non-blocking async task. Results are stored for
    dashboard display in "Portfolio Health" section.
    """

    def __init__(self, max_daily_loss_pct: float = 8.0) -> None:
        self._max_daily_loss_pct = max_daily_loss_pct
        self._last_report: StressTestReport | None = None
        self._history: list[StressTestReport] = []
        self._last_run: float = 0

    @property
    def last_report(self) -> StressTestReport | None:
        return self._last_report

    @property
    def history(self) -> list[StressTestReport]:
        return self._history

    async def run_stress_test(
        self,
        positions: list[dict],
        capital: float,
        spreads: dict[str, float] | None = None,
    ) -> StressTestReport:
        """Run all 5 stress scenarios against current positions.

        Args:
            positions: List of position dicts with market_id, size_usd,
                       direction, entry_price, current_price.
            capital: Total capital (USDC).
            spreads: Optional dict of market_id -> current spread %.

        Returns:
            StressTestReport with all scenario results.
        """
        spreads = spreads or {}
        now = datetime.now(timezone.utc).isoformat()

        scenarios = [
            self._scenario_flash_crash(positions, capital),
            self._scenario_correlation_collapse(positions, capital),
            self._scenario_dry_liquidity(positions, capital, spreads),
            self._scenario_black_swan(positions, capital),
            self._scenario_dead_api(positions, capital),
        ]

        worst_pnl = min(s.simulated_pnl for s in scenarios) if scenarios else 0.0
        worst_pct = (worst_pnl / capital * 100) if capital > 0 else 0.0

        alerts = []
        for s in scenarios:
            if abs(s.pnl_pct) > LOSS_ALERT_THRESHOLD_PCT:
                alerts.append(
                    f"STRESS ALERT: {s.scenario_name} would cause "
                    f"{s.pnl_pct:.1f}% loss (${abs(s.simulated_pnl):.0f})"
                )

        report = StressTestReport(
            timestamp=now,
            capital=capital,
            scenarios=scenarios,
            worst_case_pnl=worst_pnl,
            worst_case_pct=worst_pct,
            alerts=alerts,
        )

        self._last_report = report
        self._history.append(report)
        # Keep last 30 days of reports
        if len(self._history) > 30:
            self._history = self._history[-30:]

        self._last_run = time.time()
        logger.info(
            "stress_test_complete",
            worst_case_pct=round(worst_pct, 1),
            alerts=len(alerts),
        )
        return report

    def _scenario_flash_crash(
        self, positions: list[dict], capital: float
    ) -> ScenarioResult:
        """Scenario 1: All positions drop 15% simultaneously."""
        total_loss = 0.0
        for pos in positions:
            size = pos.get("size_usd", 0)
            direction = pos.get("direction", "BUY")
            # Flash crash hurts BUY positions, helps SELL
            if direction == "BUY":
                total_loss -= size * 0.15
            else:
                total_loss += size * 0.15

        pnl_pct = (total_loss / capital * 100) if capital > 0 else 0.0
        triggers_cb = abs(pnl_pct) >= self._max_daily_loss_pct

        return ScenarioResult(
            scenario_name="flash_crash",
            description="All positions -15% simultaneously",
            simulated_pnl=total_loss,
            pnl_pct=pnl_pct,
            triggers_circuit_breaker=triggers_cb,
            max_damage_usd=abs(total_loss),
            positions_affected=len(positions),
        )

    def _scenario_correlation_collapse(
        self, positions: list[dict], capital: float
    ) -> ScenarioResult:
        """Scenario 2: Correlated positions move in opposite directions."""
        # Simulate: half move +10%, half move -10% (worst case for any direction mix)
        total_loss = 0.0
        for i, pos in enumerate(positions):
            size = pos.get("size_usd", 0)
            direction = pos.get("direction", "BUY")
            # Alternate: even positions crash, odd positions rally
            crash = (i % 2 == 0)
            if direction == "BUY":
                total_loss += size * (-0.10 if crash else 0.05)
            else:
                total_loss += size * (0.10 if crash else -0.05)

        pnl_pct = (total_loss / capital * 100) if capital > 0 else 0.0
        triggers_cb = abs(pnl_pct) >= self._max_daily_loss_pct

        return ScenarioResult(
            scenario_name="correlation_collapse",
            description="Correlated positions diverge in opposite directions",
            simulated_pnl=total_loss,
            pnl_pct=pnl_pct,
            triggers_circuit_breaker=triggers_cb,
            max_damage_usd=abs(min(total_loss, 0)),
            positions_affected=len(positions),
        )

    def _scenario_dry_liquidity(
        self, positions: list[dict], capital: float, spreads: dict[str, float]
    ) -> ScenarioResult:
        """Scenario 3: Spreads jump to historical max (assume 15%)."""
        MAX_SPREAD = 0.15
        total_slippage = 0.0
        for pos in positions:
            size = pos.get("size_usd", 0)
            market_id = pos.get("market_id", "")
            current_spread = spreads.get(market_id, 0.03)
            # Slippage = difference between max spread and current
            extra_slippage = max(MAX_SPREAD - current_spread, 0)
            total_slippage -= size * extra_slippage

        pnl_pct = (total_slippage / capital * 100) if capital > 0 else 0.0
        triggers_cb = abs(pnl_pct) >= self._max_daily_loss_pct

        return ScenarioResult(
            scenario_name="dry_liquidity",
            description="All spreads jump to 15% (historical max)",
            simulated_pnl=total_slippage,
            pnl_pct=pnl_pct,
            triggers_circuit_breaker=triggers_cb,
            max_damage_usd=abs(total_slippage),
            positions_affected=len(positions),
        )

    def _scenario_black_swan(
        self, positions: list[dict], capital: float
    ) -> ScenarioResult:
        """Scenario 4: Largest position loses 100% (market resolves against bot)."""
        if not positions:
            return ScenarioResult(
                scenario_name="black_swan",
                description="Largest position loses 100%",
                simulated_pnl=0.0,
                pnl_pct=0.0,
                triggers_circuit_breaker=False,
                max_damage_usd=0.0,
                positions_affected=0,
            )

        # Find largest position
        largest = max(positions, key=lambda p: p.get("size_usd", 0))
        loss = -largest.get("size_usd", 0)
        pnl_pct = (loss / capital * 100) if capital > 0 else 0.0
        triggers_cb = abs(pnl_pct) >= self._max_daily_loss_pct

        return ScenarioResult(
            scenario_name="black_swan",
            description=f"Largest position ({largest.get('market_id', '?')[:20]}) loses 100%",
            simulated_pnl=loss,
            pnl_pct=pnl_pct,
            triggers_circuit_breaker=triggers_cb,
            max_damage_usd=abs(loss),
            positions_affected=1,
            details={"market_id": largest.get("market_id", ""), "size_usd": largest.get("size_usd", 0)},
        )

    def _scenario_dead_api(
        self, positions: list[dict], capital: float
    ) -> ScenarioResult:
        """Scenario 5: API dead for 4 hours — can't manage positions.

        Assume worst-case drift: positions move against us by 5% with no exit.
        """
        total_loss = 0.0
        for pos in positions:
            size = pos.get("size_usd", 0)
            # 4h unmanaged = assume 5% adverse move without stop loss
            total_loss -= size * 0.05

        pnl_pct = (total_loss / capital * 100) if capital > 0 else 0.0
        triggers_cb = abs(pnl_pct) >= self._max_daily_loss_pct

        return ScenarioResult(
            scenario_name="dead_api_4h",
            description="API dead for 4 hours, positions unmanaged",
            simulated_pnl=total_loss,
            pnl_pct=pnl_pct,
            triggers_circuit_breaker=triggers_cb,
            max_damage_usd=abs(total_loss),
            positions_affected=len(positions),
            details={"assumed_drift_pct": 5.0, "hours_unmanaged": 4},
        )

    def should_run(self) -> bool:
        """Check if stress test should run (daily at 00:00 UTC)."""
        now = datetime.now(timezone.utc)
        if now.hour != 0:
            return False
        # Don't run more than once per 20 hours
        if time.time() - self._last_run < 20 * 3600:
            return False
        return True
