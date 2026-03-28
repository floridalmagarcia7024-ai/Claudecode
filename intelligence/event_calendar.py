"""Event calendar for market-relevant events (Module 12C).

Configurable list of events with timestamps and keywords.
Blocks new positions if a relevant event is within 2 hours,
and evaluates preventive closes if event is within 30 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CalendarEvent:
    """A scheduled event that may affect markets."""

    name: str
    timestamp: str  # ISO format
    keywords: list[str] = field(default_factory=list)
    category: str = "other"
    impact: str = "medium"  # "high", "medium", "low"


class EventCalendar:
    """Manages scheduled events and their impact on trading decisions.

    Rules:
      - Event in < 2h: no new positions for affected markets
      - Event in < 30min with position in profit: evaluate preventive close
    """

    def __init__(self, events: list[CalendarEvent] | None = None) -> None:
        self._events: list[CalendarEvent] = events or []

    def add_event(self, event: CalendarEvent) -> None:
        """Add a new event to the calendar."""
        self._events.append(event)
        logger.info("event_added", name=event.name, timestamp=event.timestamp)

    def remove_past_events(self) -> int:
        """Remove events that have already passed. Returns count removed."""
        now = datetime.now(timezone.utc)
        before = len(self._events)
        self._events = [
            e for e in self._events
            if self._parse_timestamp(e.timestamp) is not None
            and self._parse_timestamp(e.timestamp) > now  # type: ignore[operator]
        ]
        removed = before - len(self._events)
        if removed:
            logger.info("past_events_removed", count=removed)
        return removed

    def should_block_new_position(self, market_keywords: list[str]) -> bool:
        """Check if a relevant event is within 2 hours.

        Args:
            market_keywords: Keywords associated with the market.

        Returns:
            True if a relevant event is imminent.
        """
        now = datetime.now(timezone.utc)
        for event in self._events:
            event_dt = self._parse_timestamp(event.timestamp)
            if event_dt is None:
                continue

            hours_until = (event_dt - now).total_seconds() / 3600
            if 0 < hours_until < 2:
                if self._keywords_overlap(event.keywords, market_keywords):
                    logger.info(
                        "event_blocks_position",
                        event=event.name,
                        hours_until=round(hours_until, 2),
                    )
                    return True
        return False

    def should_evaluate_close(self, market_keywords: list[str]) -> bool:
        """Check if a relevant event is within 30 minutes (evaluate preventive close).

        Args:
            market_keywords: Keywords associated with the market.

        Returns:
            True if should evaluate closing profitable positions.
        """
        now = datetime.now(timezone.utc)
        for event in self._events:
            event_dt = self._parse_timestamp(event.timestamp)
            if event_dt is None:
                continue

            minutes_until = (event_dt - now).total_seconds() / 60
            if 0 < minutes_until < 30:
                if self._keywords_overlap(event.keywords, market_keywords):
                    logger.info(
                        "event_evaluate_close",
                        event=event.name,
                        minutes_until=round(minutes_until, 1),
                    )
                    return True
        return False

    def get_upcoming_events(self, hours: float = 24) -> list[CalendarEvent]:
        """Get events occurring within the next N hours."""
        now = datetime.now(timezone.utc)
        upcoming = []
        for event in self._events:
            event_dt = self._parse_timestamp(event.timestamp)
            if event_dt is None:
                continue
            hours_until = (event_dt - now).total_seconds() / 3600
            if 0 < hours_until <= hours:
                upcoming.append(event)
        return upcoming

    @staticmethod
    def _keywords_overlap(event_keywords: list[str], market_keywords: list[str]) -> bool:
        """Check if any event keyword matches any market keyword."""
        event_set = {k.lower() for k in event_keywords}
        market_set = {k.lower() for k in market_keywords}
        return bool(event_set & market_set)

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime | None:
        """Parse an ISO timestamp string."""
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
