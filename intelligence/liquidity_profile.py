"""Hourly liquidity profile (Module 25B).

Builds a histogram of average spread by UTC hour for each market.
Prefers operating in the 4 best-liquidity hours of the day.
Does NOT block other hours — only lowers priority.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

TOP_HOURS_COUNT = 4


@dataclass
class HourlySpread:
    """Average spread data for a specific UTC hour."""

    hour: int
    avg_spread_pct: float
    sample_count: int
    is_preferred: bool = False


class LiquidityProfile:
    """Tracks and analyzes spread patterns by UTC hour per market.

    Identifies the 4 best-liquidity hours and provides a priority
    multiplier for trading decisions.
    """

    def __init__(self) -> None:
        # market_id -> hour (0-23) -> list of spread observations
        self._spread_data: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._preferred_hours: dict[str, list[int]] = {}
        self._max_observations_per_hour = 500

    def record_spread(self, market_id: str, spread_pct: float) -> None:
        """Record a spread observation for the current UTC hour.

        Args:
            market_id: Market identifier.
            spread_pct: Current spread as a percentage.
        """
        hour = datetime.now(timezone.utc).hour
        observations = self._spread_data[market_id][hour]
        observations.append(spread_pct)

        # Cap observations per hour
        if len(observations) > self._max_observations_per_hour:
            self._spread_data[market_id][hour] = observations[
                -self._max_observations_per_hour :
            ]

    def compute_profile(self, market_id: str) -> list[HourlySpread]:
        """Compute the hourly spread profile for a market.

        Returns:
            List of 24 HourlySpread entries (one per UTC hour), sorted by hour.
        """
        data = self._spread_data.get(market_id, {})
        if not data:
            return []

        # Compute average spread per hour
        hourly = []
        for hour in range(24):
            observations = data.get(hour, [])
            if observations:
                avg = statistics.mean(observations)
                hourly.append(
                    HourlySpread(
                        hour=hour,
                        avg_spread_pct=round(avg, 4),
                        sample_count=len(observations),
                    )
                )
            else:
                hourly.append(
                    HourlySpread(hour=hour, avg_spread_pct=float("inf"), sample_count=0)
                )

        # Identify top N hours with lowest spread
        valid_hours = [h for h in hourly if h.sample_count > 0]
        valid_hours.sort(key=lambda h: h.avg_spread_pct)
        preferred = {h.hour for h in valid_hours[:TOP_HOURS_COUNT]}

        for h in hourly:
            h.is_preferred = h.hour in preferred

        self._preferred_hours[market_id] = sorted(preferred)
        return hourly

    def get_priority_multiplier(self, market_id: str) -> float:
        """Get trading priority multiplier for current hour.

        Returns:
            1.0 if current hour is in top 4 liquidity hours.
            0.7 if not in top 4 (lower priority, not blocked).
            1.0 if insufficient data (default to normal priority).
        """
        preferred = self._preferred_hours.get(market_id)
        if not preferred:
            # Try to compute if we have data
            profile = self.compute_profile(market_id)
            preferred = self._preferred_hours.get(market_id)

        if not preferred:
            return 1.0  # No data — don't penalize

        current_hour = datetime.now(timezone.utc).hour
        if current_hour in preferred:
            return 1.0
        return 0.7

    def get_preferred_hours(self, market_id: str) -> list[int]:
        """Get the 4 preferred trading hours for a market."""
        if market_id not in self._preferred_hours:
            self.compute_profile(market_id)
        return self._preferred_hours.get(market_id, [])

    def get_profile_dict(self, market_id: str) -> dict:
        """Get full profile for dashboard display."""
        profile = self.compute_profile(market_id)
        return {
            "market_id": market_id,
            "preferred_hours": self.get_preferred_hours(market_id),
            "hourly_data": [
                {
                    "hour": h.hour,
                    "avg_spread_pct": h.avg_spread_pct
                    if h.avg_spread_pct != float("inf")
                    else None,
                    "sample_count": h.sample_count,
                    "is_preferred": h.is_preferred,
                }
                for h in profile
            ],
        }

    def get_all_profiles_summary(self) -> list[dict]:
        """Get summary of all tracked markets."""
        summaries = []
        for market_id in self._spread_data:
            preferred = self.get_preferred_hours(market_id)
            total_obs = sum(
                len(obs)
                for obs in self._spread_data[market_id].values()
            )
            summaries.append({
                "market_id": market_id,
                "preferred_hours": preferred,
                "total_observations": total_obs,
            })
        return summaries
