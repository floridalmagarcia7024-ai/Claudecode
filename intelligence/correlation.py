"""Cross-market correlation detector (Module 12B).

Tracks correlation between markets and enforces:
  - If correlation > CORRELATION_BLOCK: treat as single exposure
  - If historical correlation breaks by > 2 sigma: log divergence
  - Max correlated exposure: 25% of capital
"""

from __future__ import annotations

import statistics

import structlog

from config import settings

logger = structlog.get_logger(__name__)

MAX_CORRELATED_EXPOSURE_PCT = 0.25  # 25% of capital


class CorrelationTracker:
    """Tracks and enforces correlation-based position limits."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}  # market_id -> probability series

    def update_history(self, market_id: str, probability: float) -> None:
        """Add a new data point for a market."""
        if market_id not in self._history:
            self._history[market_id] = []
        self._history[market_id].append(probability)
        # Keep last 2000 points (~7 days at 5-min)
        if len(self._history[market_id]) > 2000:
            self._history[market_id] = self._history[market_id][-2000:]

    def get_correlation(self, market_a: str, market_b: str) -> float | None:
        """Calculate Pearson correlation between two markets.

        Returns:
            Correlation coefficient in [-1, 1], or None if insufficient data.
        """
        hist_a = self._history.get(market_a, [])
        hist_b = self._history.get(market_b, [])

        min_len = min(len(hist_a), len(hist_b))
        if min_len < 30:
            return None

        a = hist_a[-min_len:]
        b = hist_b[-min_len:]

        return self._pearson(a, b)

    def check_correlation_block(self, market_a: str, market_b: str) -> bool:
        """Check if two markets are highly correlated (should be treated as one).

        Returns:
            True if correlation > CORRELATION_BLOCK threshold.
        """
        corr = self.get_correlation(market_a, market_b)
        if corr is None:
            return False
        return abs(corr) > settings.correlation_block

    def check_divergence(self, market_a: str, market_b: str) -> bool:
        """Check if historical correlation has broken by >2 sigma.

        Compares recent correlation (last 50 points) vs historical.

        Returns:
            True if divergence detected.
        """
        hist_a = self._history.get(market_a, [])
        hist_b = self._history.get(market_b, [])

        min_len = min(len(hist_a), len(hist_b))
        if min_len < 100:
            return False

        # Historical correlation (full window)
        full_corr = self._pearson(hist_a[-min_len:], hist_b[-min_len:])
        # Recent correlation (last 50 points)
        recent_corr = self._pearson(hist_a[-50:], hist_b[-50:])

        if full_corr is None or recent_corr is None:
            return False

        # Calculate rolling correlations to get standard deviation
        window = 50
        rolling_corrs = []
        for i in range(window, min_len):
            c = self._pearson(
                hist_a[i - window:i],
                hist_b[i - window:i],
            )
            if c is not None:
                rolling_corrs.append(c)

        if len(rolling_corrs) < 10:
            return False

        mean_corr = statistics.mean(rolling_corrs)
        std_corr = statistics.stdev(rolling_corrs)

        if std_corr < 0.001:
            return False

        z = abs(recent_corr - mean_corr) / std_corr
        if z > 2.0:
            logger.info(
                "correlation_divergence_detected",
                market_a=market_a,
                market_b=market_b,
                historical_corr=round(full_corr, 3),
                recent_corr=round(recent_corr, 3),
                z_score=round(z, 2),
            )
            return True
        return False

    def get_correlated_exposure(
        self, market_id: str, active_positions: list[dict], capital: float
    ) -> float:
        """Calculate total exposure to correlated markets.

        Args:
            market_id: The new market to check.
            active_positions: List of dicts with 'market_id' and 'size_usd'.
            capital: Total capital.

        Returns:
            Total correlated exposure in USD.
        """
        total = 0.0
        for pos in active_positions:
            pos_market = pos.get("market_id", "")
            if pos_market == market_id:
                continue
            if self.check_correlation_block(market_id, pos_market):
                total += pos.get("size_usd", 0.0)
        return total

    def check_correlated_exposure_limit(
        self, market_id: str, new_size: float, active_positions: list[dict], capital: float
    ) -> bool:
        """Check if adding a new position would exceed correlated exposure limit.

        Returns:
            True if within limits.
        """
        current_exposure = self.get_correlated_exposure(market_id, active_positions, capital)
        max_exposure = capital * MAX_CORRELATED_EXPOSURE_PCT
        return (current_exposure + new_size) <= max_exposure

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2 or len(y) != n:
            return None

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = (var_x * var_y) ** 0.5
        if denom < 1e-10:
            return None

        return cov / denom
