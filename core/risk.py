"""Risk management: trailing stop, circuit breaker, anti-martingale, sizing.

Hardcoded safety limits that cannot be overridden by configuration.
"""

from __future__ import annotations

import structlog

from api.polymarket import OrderBook
from config import settings
from core.state import Position, PortfolioState, TrailingState

logger = structlog.get_logger(__name__)

# Category exposure limits — hardcoded, not user-configurable
CATEGORY_LIMITS: dict[str, float] = {
    "politics": 0.40,
    "crypto": 0.30,
    "sports": 0.20,
    "other": 0.30,
}


class RiskManager:
    """Centralized risk management for the trading bot."""

    # ── Trailing Stop State Machine ────────────────────────────
    #
    # WATCHING  → price < entry * (1 + BREAKEVEN_TRIGGER)
    # BREAKEVEN → SL = entry_price (protect capital)
    # TRAILING  → SL = max(SL_prev, price * (1 - TRAILING_PCT))
    # CLOSING   → price <= SL → execute close

    def update_trailing_stop(
        self, position: Position, current_price: float
    ) -> tuple[TrailingState, float]:
        """Advance the trailing stop state machine.

        Args:
            position: Current position with trailing state and stop loss.
            current_price: Latest market price.

        Returns:
            Tuple of (new_state, new_stop_loss).
        """
        state = position.trailing_state
        entry = position.entry_price
        sl = position.stop_loss

        # For SELL positions, invert the logic
        is_buy = position.direction == "BUY"

        if is_buy:
            profit_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0
        else:
            profit_pct = ((entry - current_price) / entry) * 100 if entry > 0 else 0

        breakeven_trigger = settings.breakeven_trigger
        trailing_pct = settings.trailing_pct

        new_state = state
        new_sl = sl

        if state == TrailingState.WATCHING:
            if profit_pct >= breakeven_trigger:
                new_state = TrailingState.BREAKEVEN
                new_sl = entry  # Protect capital
                logger.info(
                    "trailing_state_change",
                    position_id=position.id,
                    from_state="WATCHING",
                    to_state="BREAKEVEN",
                    profit_pct=round(profit_pct, 2),
                )

        elif state == TrailingState.BREAKEVEN:
            # Advance to trailing once we have enough profit
            if is_buy:
                trail_sl = current_price * (1 - trailing_pct / 100)
            else:
                trail_sl = current_price * (1 + trailing_pct / 100)

            if is_buy and trail_sl > entry:
                new_state = TrailingState.TRAILING
                new_sl = trail_sl
                logger.info(
                    "trailing_state_change",
                    position_id=position.id,
                    from_state="BREAKEVEN",
                    to_state="TRAILING",
                    new_sl=round(new_sl, 4),
                )
            elif not is_buy and trail_sl < entry:
                new_state = TrailingState.TRAILING
                new_sl = trail_sl
                logger.info(
                    "trailing_state_change",
                    position_id=position.id,
                    from_state="BREAKEVEN",
                    to_state="TRAILING",
                    new_sl=round(new_sl, 4),
                )

        elif state == TrailingState.TRAILING:
            if is_buy:
                candidate_sl = current_price * (1 - trailing_pct / 100)
                new_sl = max(sl, candidate_sl)
            else:
                candidate_sl = current_price * (1 + trailing_pct / 100)
                new_sl = min(sl, candidate_sl) if sl > 0 else candidate_sl

        # Check for CLOSING condition in any trailing/breakeven state
        if new_state in (TrailingState.BREAKEVEN, TrailingState.TRAILING) and new_sl > 0:
            if is_buy and current_price <= new_sl:
                new_state = TrailingState.CLOSING
                logger.info(
                    "trailing_stop_triggered",
                    position_id=position.id,
                    current_price=round(current_price, 4),
                    stop_loss=round(new_sl, 4),
                    direction=position.direction,
                )
            elif not is_buy and current_price >= new_sl:
                new_state = TrailingState.CLOSING
                logger.info(
                    "trailing_stop_triggered",
                    position_id=position.id,
                    current_price=round(current_price, 4),
                    stop_loss=round(new_sl, 4),
                    direction=position.direction,
                )

        return new_state, new_sl

    # ── Circuit Breaker ────────────────────────────────────────

    def check_circuit_breaker(self, daily_pnl: float, capital: float) -> bool:
        """Check if daily loss exceeds the maximum allowed.

        Args:
            daily_pnl: Total P&L for today (negative = loss).
            capital: Current total capital.

        Returns:
            True if circuit breaker is TRIGGERED (should stop trading).
        """
        if capital <= 0:
            return True

        loss_pct = abs(daily_pnl) / capital * 100 if daily_pnl < 0 else 0
        triggered = loss_pct > settings.max_daily_loss_pct

        if triggered:
            logger.warning(
                "circuit_breaker_triggered",
                daily_pnl=round(daily_pnl, 2),
                loss_pct=round(loss_pct, 2),
                max_allowed=settings.max_daily_loss_pct,
                capital=round(capital, 2),
            )

        return triggered

    # ── Position Sizing ────────────────────────────────────────

    def calculate_position_size(
        self, portfolio: PortfolioState, category: str = "other"
    ) -> float:
        """Calculate position size using fixed % or Quarter Kelly.

        Rules:
            < 50 trades: DEFAULT_POSITION_PCT (fixed)
            >= 50 trades: Quarter Kelly criterion

        Then applies anti-martingale reduction and category limits.

        Args:
            portfolio: Current portfolio state.
            category: Market category for limit checks.

        Returns:
            Position size in USD.
        """
        capital = portfolio.capital

        if portfolio.total_trades < 50:
            base_size = capital * (settings.default_position_pct / 100)
        else:
            kelly = self._quarter_kelly(
                portfolio.win_rate, portfolio.avg_win, portfolio.avg_loss
            )
            base_size = capital * min(kelly, settings.max_position_pct / 100)

        # Apply anti-martingale
        size = self.apply_anti_martingale(base_size, portfolio.consecutive_losses)

        # Check category limits
        if not self.check_category_limit(category, portfolio, size):
            remaining = self._remaining_category_capacity(category, portfolio)
            size = max(0, min(size, remaining))
            if size <= 0:
                logger.info(
                    "size_rejected_category_limit",
                    category=category,
                    limit=CATEGORY_LIMITS.get(category, 0.30),
                )
                return 0.0

        # Enforce maximum
        max_size = capital * (settings.max_position_pct / 100)
        size = min(size, max_size)

        return round(size, 2)

    def apply_anti_martingale(self, base_size: float, consecutive_losses: int) -> float:
        """Reduce position size after consecutive losses.

        recovery_factor = max(0.30, 1.0 - consecutive_losses * 0.15)
        If consecutive_losses >= 5: PAUSE (return 0).

        Args:
            base_size: Pre-adjustment position size.
            consecutive_losses: Number of consecutive losses.

        Returns:
            Adjusted position size.
        """
        if consecutive_losses >= 5:
            logger.warning(
                "anti_martingale_pause",
                consecutive_losses=consecutive_losses,
                msg="5+ consecutive losses — 24h trading pause",
            )
            return 0.0

        recovery_factor = max(0.30, 1.0 - consecutive_losses * 0.15)
        adjusted = base_size * recovery_factor

        if consecutive_losses > 0:
            logger.info(
                "anti_martingale_applied",
                consecutive_losses=consecutive_losses,
                recovery_factor=round(recovery_factor, 2),
                original_size=round(base_size, 2),
                adjusted_size=round(adjusted, 2),
            )

        return adjusted

    def check_category_limit(
        self, category: str, portfolio: PortfolioState, new_size: float
    ) -> bool:
        """Check if adding a position would exceed category exposure limits.

        Args:
            category: Market category.
            portfolio: Current portfolio state.
            new_size: Proposed position size in USD.

        Returns:
            True if within limits.
        """
        limit = CATEGORY_LIMITS.get(category, 0.30)
        current_exposure = portfolio.category_exposure.get(category, 0.0)
        max_exposure = portfolio.capital * limit

        return (current_exposure + new_size) <= max_exposure

    # ── Liquidity Filter ───────────────────────────────────────

    def check_liquidity(
        self, orderbook: OrderBook, position_size_usd: float
    ) -> tuple[bool, str]:
        """Pre-trade liquidity check.

        Checks:
            1. spread_pct <= MAX_SPREAD_PCT
            2. depth_5_usd >= MIN_DEPTH_USD
            3. slippage_est <= MAX_SLIPPAGE_PCT

        Args:
            orderbook: Current order book snapshot.
            position_size_usd: Proposed position size.

        Returns:
            Tuple of (passes, rejection_reason).
        """
        # Spread check
        if orderbook.spread_pct > settings.max_spread_pct:
            reason = (
                f"spread_too_wide: {orderbook.spread_pct:.2f}% > {settings.max_spread_pct}%"
            )
            logger.info(
                "liquidity_rejected",
                reason="spread_too_wide",
                spread_pct=round(orderbook.spread_pct, 2),
                max_spread=settings.max_spread_pct,
            )
            return False, reason

        # Depth check
        if orderbook.depth_5_usd < settings.min_depth_usd:
            reason = (
                f"insufficient_depth: ${orderbook.depth_5_usd:.0f} < ${settings.min_depth_usd}"
            )
            logger.info(
                "liquidity_rejected",
                reason="insufficient_depth",
                depth_usd=round(orderbook.depth_5_usd, 2),
                min_depth=settings.min_depth_usd,
            )
            return False, reason

        # Slippage estimate
        if orderbook.depth_5_usd > 0:
            slippage_est = position_size_usd / orderbook.depth_5_usd * 100
        else:
            slippage_est = 100.0

        if slippage_est > settings.max_slippage_pct:
            reason = (
                f"slippage_too_high: {slippage_est:.2f}% > {settings.max_slippage_pct}%"
            )
            logger.info(
                "liquidity_rejected",
                reason="slippage_too_high",
                slippage_est=round(slippage_est, 2),
                max_slippage=settings.max_slippage_pct,
                position_size=position_size_usd,
                depth_usd=round(orderbook.depth_5_usd, 2),
            )
            return False, reason

        return True, ""

    # ── Internal ───────────────────────────────────────────────

    @staticmethod
    def _quarter_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Quarter Kelly criterion.

        kelly_full = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = kelly_full * 0.25

        Returns:
            Kelly fraction as a decimal (e.g., 0.05 for 5%).
        """
        if avg_win <= 0 or win_rate <= 0:
            return settings.default_position_pct / 100

        kelly_full = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        if kelly_full <= 0:
            return settings.default_position_pct / 100

        return kelly_full * 0.25

    def _remaining_category_capacity(
        self, category: str, portfolio: PortfolioState
    ) -> float:
        """Calculate remaining USD capacity for a category."""
        limit = CATEGORY_LIMITS.get(category, 0.30)
        current = portfolio.category_exposure.get(category, 0.0)
        max_exposure = portfolio.capital * limit
        return max(0, max_exposure - current)
