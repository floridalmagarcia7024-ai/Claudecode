"""Order flow monitor + manipulation detection (Module 25A + 25C).

Sub-module 25A — Order Flow Monitor (every 30s):
  Detects large orders (> $2,000) in the order book.
  Records direction → "whale_pressure".
  If whale confirms signal → boost confidence +0.10
  If contradicts → reduce confidence -0.15

Sub-module 25C — Manipulation Detection:
  Stop hunting: price touches SL and rebounds > 2% in 1 min → log suspicion
  Wash trading: many transactions, no real price movement → reduce volume confidence
  Spoofing: large order appears and disappears < 10s → ignore signal
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

WHALE_ORDER_THRESHOLD_USD = 2000.0
SPOOF_DISAPPEAR_SECONDS = 10
STOP_HUNT_REBOUND_PCT = 0.02


@dataclass
class WhaleOrder:
    """A detected large order in the book."""

    market_id: str
    side: str  # "BUY" or "SELL"
    size_usd: float
    price: float
    timestamp: float = 0.0
    still_present: bool = True


@dataclass
class ManipulationEvent:
    """A detected manipulation suspicion."""

    market_id: str
    event_type: str  # "stop_hunting", "wash_trading", "spoofing"
    description: str
    timestamp: str = ""
    severity: str = "medium"  # low, medium, high

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "event_type": self.event_type,
            "description": self.description,
            "timestamp": self.timestamp,
            "severity": self.severity,
        }


class OrderFlowMonitor:
    """Monitors order flow for whale activity and manipulation signals.

    Provides confidence adjustments based on whale pressure direction.
    """

    def __init__(self) -> None:
        # whale_pressure per market: positive = buy pressure, negative = sell
        self._whale_pressure: dict[str, float] = defaultdict(float)
        self._whale_orders: dict[str, list[WhaleOrder]] = defaultdict(list)
        self._manipulation_events: list[ManipulationEvent] = []
        self._price_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # Track previous orderbook snapshots for spoof detection
        self._prev_large_orders: dict[str, list[WhaleOrder]] = defaultdict(list)
        self._volume_confidence: dict[str, float] = defaultdict(lambda: 1.0)

    def analyze_orderbook(
        self,
        market_id: str,
        bids: list[tuple[float, float]],  # (price, size_usd)
        asks: list[tuple[float, float]],
    ) -> dict[str, float]:
        """Analyze an orderbook snapshot for whale orders.

        Args:
            market_id: The market identifier.
            bids: List of (price, size_usd) for bid side.
            asks: List of (price, size_usd) for ask side.

        Returns:
            Dict with whale_pressure and any confidence adjustment.
        """
        now = time.time()
        current_whales = []

        # Detect large buy orders
        for price, size in bids:
            if size >= WHALE_ORDER_THRESHOLD_USD:
                whale = WhaleOrder(
                    market_id=market_id,
                    side="BUY",
                    size_usd=size,
                    price=price,
                    timestamp=now,
                )
                current_whales.append(whale)
                self._whale_pressure[market_id] += size

        # Detect large sell orders
        for price, size in asks:
            if size >= WHALE_ORDER_THRESHOLD_USD:
                whale = WhaleOrder(
                    market_id=market_id,
                    side="SELL",
                    size_usd=size,
                    price=price,
                    timestamp=now,
                )
                current_whales.append(whale)
                self._whale_pressure[market_id] -= size

        # Spoof detection: check if previous large orders disappeared quickly
        self._detect_spoofing(market_id, current_whales, now)

        # Update tracking
        self._prev_large_orders[market_id] = current_whales
        self._whale_orders[market_id] = (
            self._whale_orders[market_id][-100:] + current_whales
        )

        pressure = self._whale_pressure.get(market_id, 0.0)
        return {
            "whale_pressure": pressure,
            "large_orders_count": len(current_whales),
            "volume_confidence": self._volume_confidence.get(market_id, 1.0),
        }

    def get_confidence_adjustment(
        self, market_id: str, signal_direction: str
    ) -> float:
        """Get confidence adjustment based on whale pressure.

        Returns:
            +0.10 if whale pressure confirms signal direction.
            -0.15 if whale pressure contradicts signal direction.
            0.0 if neutral or no data.
        """
        pressure = self._whale_pressure.get(market_id, 0.0)

        # Normalize pressure — need meaningful threshold
        if abs(pressure) < WHALE_ORDER_THRESHOLD_USD:
            return 0.0

        if signal_direction == "BUY" and pressure > 0:
            return 0.10
        elif signal_direction == "BUY" and pressure < 0:
            return -0.15
        elif signal_direction == "SELL" and pressure < 0:
            return 0.10
        elif signal_direction == "SELL" and pressure > 0:
            return -0.15

        return 0.0

    def record_price(self, market_id: str, price: float) -> None:
        """Record a price tick for manipulation detection."""
        now = time.time()
        history = self._price_history[market_id]
        history.append((now, price))
        # Keep last 500 ticks
        if len(history) > 500:
            self._price_history[market_id] = history[-500:]

    def detect_stop_hunting(
        self, market_id: str, stop_loss_price: float, current_price: float
    ) -> ManipulationEvent | None:
        """Detect stop hunting: price touches SL and rebounds > 2% in 1 min.

        Args:
            market_id: Market to check.
            stop_loss_price: The stop loss level.
            current_price: Current market price.

        Returns:
            ManipulationEvent if suspicious, None otherwise.
        """
        history = self._price_history.get(market_id, [])
        if len(history) < 2:
            return None

        now = time.time()

        # Check if SL was touched in last 60 seconds
        sl_touched = False
        for ts, price in reversed(history):
            if now - ts > 60:
                break
            if abs(price - stop_loss_price) / stop_loss_price < 0.005:
                sl_touched = True
                break

        if not sl_touched:
            return None

        # Check for rebound > 2%
        rebound = abs(current_price - stop_loss_price) / stop_loss_price
        if rebound >= STOP_HUNT_REBOUND_PCT:
            event = ManipulationEvent(
                market_id=market_id,
                event_type="stop_hunting",
                description=(
                    f"Price touched SL at {stop_loss_price:.4f} and rebounded "
                    f"{rebound*100:.1f}% to {current_price:.4f} within 1 min"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="high",
            )
            self._manipulation_events.append(event)
            logger.warning("stop_hunting_detected", market_id=market_id, rebound_pct=round(rebound * 100, 1))
            return event

        return None

    def detect_wash_trading(
        self, market_id: str, trade_count: int, price_change_pct: float
    ) -> ManipulationEvent | None:
        """Detect wash trading: many trades with minimal price movement.

        Args:
            market_id: Market to check.
            trade_count: Number of recent trades.
            price_change_pct: Net price change in period.

        Returns:
            ManipulationEvent if suspicious, None otherwise.
        """
        if trade_count < 20:
            return None

        if abs(price_change_pct) < 0.5 and trade_count > 50:
            # Many trades, no movement → suspicious
            self._volume_confidence[market_id] = max(
                self._volume_confidence.get(market_id, 1.0) - 0.20, 0.2
            )
            event = ManipulationEvent(
                market_id=market_id,
                event_type="wash_trading",
                description=(
                    f"{trade_count} trades with only {price_change_pct:.2f}% "
                    f"price change — volume confidence reduced"
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="medium",
            )
            self._manipulation_events.append(event)
            logger.warning("wash_trading_suspected", market_id=market_id, trades=trade_count)
            return event

        return None

    def _detect_spoofing(
        self, market_id: str, current_whales: list[WhaleOrder], now: float
    ) -> None:
        """Detect spoofing: large order appears and disappears < 10s."""
        prev = self._prev_large_orders.get(market_id, [])
        if not prev:
            return

        for prev_whale in prev:
            # Check if this whale order is still present
            still_present = any(
                abs(w.price - prev_whale.price) < 0.001
                and w.side == prev_whale.side
                for w in current_whales
            )
            if not still_present and (now - prev_whale.timestamp) < SPOOF_DISAPPEAR_SECONDS:
                event = ManipulationEvent(
                    market_id=market_id,
                    event_type="spoofing",
                    description=(
                        f"Large {prev_whale.side} order ${prev_whale.size_usd:.0f} "
                        f"@ {prev_whale.price:.4f} disappeared in "
                        f"{now - prev_whale.timestamp:.1f}s"
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity="high",
                )
                self._manipulation_events.append(event)
                logger.warning(
                    "spoofing_detected",
                    market_id=market_id,
                    side=prev_whale.side,
                    size=prev_whale.size_usd,
                )

    def get_manipulation_events(self, limit: int = 20) -> list[dict]:
        """Get recent manipulation events for dashboard."""
        return [e.to_dict() for e in self._manipulation_events[-limit:]]

    def get_whale_summary(self) -> dict[str, float]:
        """Get whale pressure summary per market."""
        return dict(self._whale_pressure)

    def reset_pressure(self, market_id: str) -> None:
        """Reset whale pressure for a market (e.g., after trade execution)."""
        self._whale_pressure[market_id] = 0.0
