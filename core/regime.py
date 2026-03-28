"""Market regime detector (Module 10).

Classifies each market into one of 4 regimes every 4 hours:
  TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY.

Uses scipy.stats.linregress for proper slope calculation.
"""

from __future__ import annotations

import enum
import time
from datetime import datetime, timedelta, timezone

import structlog
from scipy import stats

logger = structlog.get_logger(__name__)

# Detection runs every 4 hours
REGIME_CHECK_INTERVAL_S = 4 * 3600
# Minimum data points needed (7 days at 5-min intervals)
MIN_DATA_POINTS_REGIME = 168  # 12h of 5-min data at minimum for 12h window


class MarketRegime(str, enum.Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class MarketRegimeDetector:
    """Detects market regime from historical probability data.

    If insufficient data (<7 days), defaults to RANGING.
    """

    def __init__(self) -> None:
        self._regimes: dict[str, MarketRegime] = {}
        self._last_check: dict[str, float] = {}

    def get_regime(self, market_id: str) -> MarketRegime:
        """Get the current regime for a market."""
        return self._regimes.get(market_id, MarketRegime.RANGING)

    def should_check(self, market_id: str) -> bool:
        """Whether it's time to re-evaluate the regime for a market."""
        last = self._last_check.get(market_id, 0)
        return (time.monotonic() - last) >= REGIME_CHECK_INTERVAL_S

    def detect_regime(
        self, market_id: str, history: list[float], atr_history_30d: list[float] | None = None
    ) -> MarketRegime:
        """Detect regime from probability history.

        Args:
            market_id: Market identifier.
            history: Probability series (most recent last), ideally 12h+ of data.
            atr_history_30d: ATR(14) values over last 30 days for volatility check.

        Returns:
            Detected MarketRegime.
        """
        self._last_check[market_id] = time.monotonic()
        old_regime = self._regimes.get(market_id)

        if len(history) < MIN_DATA_POINTS_REGIME:
            regime = MarketRegime.RANGING
            self._regimes[market_id] = regime
            if old_regime and old_regime != regime:
                logger.info("regime_changed", market_id=market_id, from_regime=old_regime, to_regime=regime.value, reason="insufficient_data")
            return regime

        # Use last 12h of data (144 points at 5-min intervals)
        window_12h = history[-144:] if len(history) >= 144 else history

        # Check HIGH_VOLATILITY first (takes priority)
        if atr_history_30d and len(atr_history_30d) >= 30:
            current_atr = atr_history_30d[-1]
            avg_atr_30d = sum(atr_history_30d) / len(atr_history_30d)
            if avg_atr_30d > 0 and current_atr > 2 * avg_atr_30d:
                regime = MarketRegime.HIGH_VOLATILITY
                self._regimes[market_id] = regime
                if old_regime != regime:
                    logger.info("regime_changed", market_id=market_id, from_regime=old_regime, to_regime=regime.value, atr_ratio=round(current_atr / avg_atr_30d, 2))
                return regime

        # Calculate hourly changes (every 12 points at 5-min intervals)
        hourly_changes = []
        step = 12  # 12 * 5min = 1 hour
        for i in range(step, len(window_12h), step):
            change = window_12h[i] - window_12h[i - step]
            hourly_changes.append(change)

        if not hourly_changes:
            regime = MarketRegime.RANGING
            self._regimes[market_id] = regime
            return regime

        # Calculate slope using scipy linregress (per hour)
        x_hours = list(range(len(window_12h)))
        slope, _, _, _, _ = stats.linregress(x_hours, window_12h)
        # Convert slope from per-data-point to per-hour (12 points per hour)
        slope_per_hour = slope * step

        positive_pct = sum(1 for c in hourly_changes if c > 0) / len(hourly_changes)
        negative_pct = sum(1 for c in hourly_changes if c < 0) / len(hourly_changes)

        # Check TRENDING_UP
        if positive_pct > 0.60 and slope_per_hour > 0.002:
            regime = MarketRegime.TRENDING_UP
        # Check TRENDING_DOWN
        elif negative_pct > 0.60 and slope_per_hour < -0.002:
            regime = MarketRegime.TRENDING_DOWN
        # Check RANGING: price oscillates ±2% around mean, BB width < 0.04
        elif self._is_ranging(window_12h):
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.RANGING  # Default

        self._regimes[market_id] = regime
        if old_regime and old_regime != regime:
            logger.info(
                "regime_changed",
                market_id=market_id,
                from_regime=old_regime,
                to_regime=regime.value,
                slope_per_hour=round(slope_per_hour, 6),
                positive_pct=round(positive_pct, 2),
            )
        return regime

    @staticmethod
    def _is_ranging(window: list[float]) -> bool:
        """Check if price oscillates ±2% around mean with tight Bollinger Bands."""
        if len(window) < 20:
            return True

        mean = sum(window) / len(window)
        if mean <= 0:
            return True

        # Check ±2% band
        for val in window:
            if abs(val - mean) / mean > 0.02:
                return False

        # Bollinger Band width: 2 * std / mean
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        std = variance ** 0.5
        bb_width = 2 * std / mean if mean > 0 else 0

        return bb_width < 0.04

    @staticmethod
    def compute_atr(closes: list[float], period: int = 14) -> list[float]:
        """Compute ATR(period) from close prices.

        For prediction markets (no high/low), uses absolute close-to-close changes.

        Returns:
            List of ATR values (one per day if daily data).
        """
        if len(closes) < period + 1:
            return []

        true_ranges = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]

        atrs = []
        # First ATR is simple average
        first_atr = sum(true_ranges[:period]) / period
        atrs.append(first_atr)

        # Subsequent ATRs use exponential smoothing
        for i in range(period, len(true_ranges)):
            atr = (atrs[-1] * (period - 1) + true_ranges[i]) / period
            atrs.append(atr)

        return atrs
