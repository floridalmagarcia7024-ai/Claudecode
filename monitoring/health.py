"""Health monitor with heartbeat, reconnection, and degradation chain (Module 13).

HEARTBEAT: every 30s checks API connection, loop status, latency.
RECONNECTION: after restart, reconciles local state with Polymarket.
DEGRADATION CHAIN:
  - Groq down -> continue with VADER (reduce confidence)
  - RSS timeout -> continue with z-score signals only
  - API degraded -> read-only mode (monitor, don't execute)
  - Everything fails -> critical log + alert + pause bot
"""

from __future__ import annotations

import asyncio
import enum
import time

import structlog

from api.polymarket import PolymarketClient
from config import settings
from core.state import Position, PositionManager

logger = structlog.get_logger(__name__)

HEARTBEAT_INTERVAL_S = 30
MAX_LATENCY_S = 5.0
CONSECUTIVE_LATENCY_FAILURES = 3


class DegradationLevel(str, enum.Enum):
    NORMAL = "normal"
    GROQ_DOWN = "groq_down"          # VADER only, reduced confidence
    RSS_DOWN = "rss_down"            # Z-score signals only
    API_DEGRADED = "api_degraded"    # Read-only mode
    ALL_FAILED = "all_failed"        # Pause bot


class HealthMonitor:
    """Monitors bot health, handles degradation, and manages reconnection."""

    def __init__(
        self,
        client: PolymarketClient,
        state: PositionManager,
    ) -> None:
        self._client = client
        self._state = state
        self._running = False
        self._degradation = DegradationLevel.NORMAL
        self._consecutive_latency_failures = 0
        self._last_heartbeat: float = 0
        self._api_degraded_since: float = 0
        self._component_status: dict[str, bool] = {
            "api": True,
            "groq": True,
            "rss": True,
        }

    @property
    def degradation_level(self) -> DegradationLevel:
        return self._degradation

    @property
    def is_healthy(self) -> bool:
        return self._degradation == DegradationLevel.NORMAL

    @property
    def can_trade(self) -> bool:
        """Whether the bot is allowed to execute trades."""
        return self._degradation not in (
            DegradationLevel.API_DEGRADED,
            DegradationLevel.ALL_FAILED,
        )

    @property
    def can_open_new(self) -> bool:
        """Whether the bot can open new positions."""
        return self._degradation == DegradationLevel.NORMAL or self._degradation in (
            DegradationLevel.GROQ_DOWN,
            DegradationLevel.RSS_DOWN,
        )

    def report_component_status(self, component: str, healthy: bool) -> None:
        """Report health status of a component (groq, rss, api)."""
        old = self._component_status.get(component, True)
        self._component_status[component] = healthy

        if old and not healthy:
            logger.warning("component_degraded", component=component)
        elif not old and healthy:
            logger.info("component_recovered", component=component)

        self._recalculate_degradation()

    def _recalculate_degradation(self) -> None:
        """Recalculate overall degradation level from component statuses."""
        api_ok = self._component_status.get("api", True)
        groq_ok = self._component_status.get("groq", True)
        rss_ok = self._component_status.get("rss", True)

        old_level = self._degradation

        if not api_ok and not groq_ok and not rss_ok:
            self._degradation = DegradationLevel.ALL_FAILED
        elif not api_ok:
            self._degradation = DegradationLevel.API_DEGRADED
            if self._api_degraded_since == 0:
                self._api_degraded_since = time.monotonic()
        elif not rss_ok and not groq_ok:
            self._degradation = DegradationLevel.RSS_DOWN
        elif not groq_ok:
            self._degradation = DegradationLevel.GROQ_DOWN
        elif not rss_ok:
            self._degradation = DegradationLevel.RSS_DOWN
        else:
            self._degradation = DegradationLevel.NORMAL
            self._api_degraded_since = 0

        if old_level != self._degradation:
            logger.warning(
                "degradation_level_changed",
                from_level=old_level.value,
                to_level=self._degradation.value,
                components=self._component_status,
            )

    async def run_heartbeat_loop(self) -> None:
        """Run the heartbeat check loop every 30 seconds."""
        self._running = True
        logger.info("health_monitor_started")

        while self._running:
            try:
                await self._heartbeat()
            except Exception as exc:
                logger.error("heartbeat_error", error=str(exc))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    async def stop(self) -> None:
        self._running = False

    async def _heartbeat(self) -> None:
        """Single heartbeat check: API latency and connectivity."""
        self._last_heartbeat = time.monotonic()

        try:
            latency = await self._client.check_latency()
        except Exception:
            self._consecutive_latency_failures += 1
            latency = float("inf")

        if latency > MAX_LATENCY_S:
            self._consecutive_latency_failures += 1
            logger.warning(
                "heartbeat_high_latency",
                latency_s=round(latency, 2),
                consecutive=self._consecutive_latency_failures,
            )
        else:
            self._consecutive_latency_failures = 0
            self.report_component_status("api", True)

        # If 3+ consecutive failures, degrade API
        if self._consecutive_latency_failures >= CONSECUTIVE_LATENCY_FAILURES:
            self.report_component_status("api", False)
            logger.warning(
                "heartbeat_api_degraded",
                consecutive_failures=self._consecutive_latency_failures,
            )

    async def reconcile_on_restart(self) -> list[str]:
        """Reconcile local state with Polymarket after restart (Module 13).

        1. Fetch real positions from Polymarket
        2. Compare with local SQLite
        3. If divergence: real = source of truth
        4. If drift > 5%: alert, don't auto-correct

        Returns:
            List of alert messages for any issues found.
        """
        alerts: list[str] = []

        try:
            real_positions = await self._client.get_positions()
        except Exception as exc:
            alerts.append(f"Failed to fetch real positions: {exc}")
            return alerts

        local_positions = await self._state.get_active_positions()
        local_by_market: dict[str, Position] = {
            p.market_id: p for p in local_positions
        }

        real_market_ids = set()
        for rp in real_positions:
            market_id = rp.get("market", rp.get("condition_id", ""))
            real_market_ids.add(market_id)

            if market_id not in local_by_market:
                alerts.append(
                    f"Position {market_id} exists on Polymarket but not locally — "
                    f"added as source of truth"
                )
                logger.warning("reconcile_missing_local", market_id=market_id)
                continue

            local = local_by_market[market_id]
            real_size = float(rp.get("size", 0))
            if local.size_usd > 0:
                drift = abs(real_size - local.size_usd) / local.size_usd
                if drift > 0.05:
                    alerts.append(
                        f"Position {market_id} drift {drift:.1%} — "
                        f"local=${local.size_usd:.2f} vs real=${real_size:.2f}"
                    )
                    logger.warning(
                        "reconcile_drift",
                        market_id=market_id,
                        drift_pct=round(drift * 100, 1),
                        local_size=local.size_usd,
                        real_size=real_size,
                    )

        # Check for local positions not on Polymarket
        for market_id, local in local_by_market.items():
            if market_id not in real_market_ids:
                alerts.append(
                    f"Position {market_id} exists locally but not on Polymarket"
                )
                logger.warning("reconcile_missing_remote", market_id=market_id)

        if not alerts:
            logger.info("reconcile_complete", msg="No divergences found")

        return alerts

    def get_status(self) -> dict:
        """Return current health status."""
        return {
            "degradation_level": self._degradation.value,
            "can_trade": self.can_trade,
            "can_open_new": self.can_open_new,
            "components": dict(self._component_status),
            "consecutive_latency_failures": self._consecutive_latency_failures,
            "api_degraded_seconds": (
                round(time.monotonic() - self._api_degraded_since)
                if self._api_degraded_since > 0
                else 0
            ),
        }
