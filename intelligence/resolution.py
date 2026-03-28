"""Resolution imminence detector (Module 12A).

Adjusts trading behavior as a market approaches its resolution date:
  < 48h  -> conservative mode: size *= 0.5
  < 4h   -> no new positions
  < 0h   -> close active positions immediately
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


class ResolutionUrgency(str, enum.Enum):
    NORMAL = "normal"           # > 48h
    CONSERVATIVE = "conservative"  # < 48h
    NO_NEW = "no_new"           # < 4h
    CLOSE_NOW = "close_now"     # < 0h (past due)


class ResolutionDetector:
    """Detects how close a market is to resolution and adjusts behavior."""

    def get_urgency(self, end_date: str) -> ResolutionUrgency:
        """Determine urgency level based on time until resolution.

        Args:
            end_date: ISO format end date of the market.

        Returns:
            ResolutionUrgency level.
        """
        hours = self.hours_until_resolution(end_date)
        if hours is None:
            return ResolutionUrgency.NORMAL

        if hours <= 0:
            return ResolutionUrgency.CLOSE_NOW
        elif hours < 4:
            return ResolutionUrgency.NO_NEW
        elif hours < 48:
            return ResolutionUrgency.CONSERVATIVE
        return ResolutionUrgency.NORMAL

    def size_multiplier(self, end_date: str) -> float:
        """Get position size multiplier based on resolution proximity.

        Returns:
            1.0 for normal, 0.5 for conservative, 0.0 for no_new/close_now.
        """
        urgency = self.get_urgency(end_date)
        if urgency == ResolutionUrgency.CONSERVATIVE:
            return 0.5
        elif urgency in (ResolutionUrgency.NO_NEW, ResolutionUrgency.CLOSE_NOW):
            return 0.0
        return 1.0

    def should_close_immediately(self, end_date: str) -> bool:
        """Whether the position should be closed immediately."""
        return self.get_urgency(end_date) == ResolutionUrgency.CLOSE_NOW

    def should_block_new_positions(self, end_date: str) -> bool:
        """Whether new positions should be blocked."""
        urgency = self.get_urgency(end_date)
        return urgency in (ResolutionUrgency.NO_NEW, ResolutionUrgency.CLOSE_NOW)

    @staticmethod
    def hours_until_resolution(end_date: str) -> float | None:
        """Calculate hours until market resolution.

        Args:
            end_date: ISO format date string.

        Returns:
            Hours until resolution, or None if date is invalid/empty.
        """
        if not end_date:
            return None
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            delta = end_dt - datetime.now(timezone.utc)
            return delta.total_seconds() / 3600
        except (ValueError, TypeError):
            return None

    def check_position_alert(self, market_id: str, end_date: str) -> str | None:
        """Generate alert message if position is near resolution.

        Returns:
            Alert message string, or None if no alert needed.
        """
        hours = self.hours_until_resolution(end_date)
        if hours is None:
            return None

        urgency = self.get_urgency(end_date)
        if urgency == ResolutionUrgency.CLOSE_NOW:
            return f"{market_id} resolves in {hours:.1f}h — CLOSE POSITION IMMEDIATELY"
        elif urgency == ResolutionUrgency.NO_NEW:
            return f"{market_id} resolves in {hours:.1f}h with active position — NO NEW TRADES"
        elif urgency == ResolutionUrgency.CONSERVATIVE:
            return f"{market_id} resolves in {hours:.1f}h with active position — conservative mode"
        return None
