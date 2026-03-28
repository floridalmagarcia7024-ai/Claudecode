"""Tests for risk management: trailing stop, circuit breaker, anti-martingale, liquidity."""

from __future__ import annotations

import pytest

from api.polymarket import OrderBook, OrderBookLevel
from core.risk import CATEGORY_LIMITS, RiskManager
from core.state import Position, PortfolioState, TrailingState


@pytest.fixture
def risk() -> RiskManager:
    return RiskManager()


class TestTrailingStop:
    """Test the 4-state trailing stop machine."""

    def _make_position(
        self,
        entry: float = 0.50,
        current: float = 0.50,
        direction: str = "BUY",
        state: TrailingState = TrailingState.WATCHING,
        sl: float = 0.0,
    ) -> Position:
        return Position(
            id=1,
            market_id="test",
            direction=direction,
            entry_price=entry,
            current_price=current,
            size_usd=100,
            stop_loss=sl,
            trailing_state=state,
        )

    def test_watching_stays_when_no_profit(self, risk: RiskManager):
        pos = self._make_position(entry=0.50, current=0.50)
        state, sl = risk.update_trailing_stop(pos, current_price=0.51)
        assert state == TrailingState.WATCHING
        assert sl == 0.0

    def test_watching_to_breakeven(self, risk: RiskManager):
        """Trigger breakeven at BREAKEVEN_TRIGGER (5%) profit."""
        pos = self._make_position(entry=0.50)
        # 5% above entry = 0.525
        state, sl = risk.update_trailing_stop(pos, current_price=0.525)
        assert state == TrailingState.BREAKEVEN
        assert sl == 0.50  # SL = entry price

    def test_breakeven_to_trailing(self, risk: RiskManager):
        """Advance to trailing when trail SL exceeds entry."""
        pos = self._make_position(
            entry=0.50, state=TrailingState.BREAKEVEN, sl=0.50
        )
        # At 0.60, trail SL = 0.60 * 0.95 = 0.57 > entry (0.50)
        state, sl = risk.update_trailing_stop(pos, current_price=0.60)
        assert state == TrailingState.TRAILING
        assert sl == pytest.approx(0.57, abs=0.01)

    def test_trailing_ratchets_up(self, risk: RiskManager):
        pos = self._make_position(
            entry=0.50, state=TrailingState.TRAILING, sl=0.57
        )
        # Price goes higher → SL should increase
        state, sl = risk.update_trailing_stop(pos, current_price=0.70)
        assert state == TrailingState.TRAILING
        assert sl > 0.57  # Ratcheted up

    def test_trailing_never_decreases(self, risk: RiskManager):
        pos = self._make_position(
            entry=0.50, state=TrailingState.TRAILING, sl=0.60
        )
        # Price drops but SL stays
        state, sl = risk.update_trailing_stop(pos, current_price=0.62)
        assert sl >= 0.60

    def test_trailing_triggers_closing(self, risk: RiskManager):
        pos = self._make_position(
            entry=0.50, state=TrailingState.TRAILING, sl=0.60
        )
        # Price drops below SL
        state, sl = risk.update_trailing_stop(pos, current_price=0.59)
        assert state == TrailingState.CLOSING

    def test_sell_direction_trailing(self, risk: RiskManager):
        """Trailing stop works in reverse for SELL positions."""
        pos = self._make_position(
            entry=0.50, direction="SELL", state=TrailingState.WATCHING
        )
        # Price drops 5%+ → breakeven
        state, sl = risk.update_trailing_stop(pos, current_price=0.475)
        assert state == TrailingState.BREAKEVEN
        assert sl == 0.50


class TestCircuitBreaker:
    def test_no_trigger_when_profitable(self, risk: RiskManager):
        assert risk.check_circuit_breaker(daily_pnl=100.0, capital=10000) is False

    def test_no_trigger_within_limit(self, risk: RiskManager):
        # 5% loss with 8% limit → ok
        assert risk.check_circuit_breaker(daily_pnl=-500.0, capital=10000) is False

    def test_triggers_at_limit(self, risk: RiskManager):
        # 9% loss with 8% limit → trigger
        assert risk.check_circuit_breaker(daily_pnl=-900.0, capital=10000) is True

    def test_triggers_zero_capital(self, risk: RiskManager):
        assert risk.check_circuit_breaker(daily_pnl=-1.0, capital=0) is True


class TestPositionSizing:
    def test_default_sizing_under_50_trades(self, risk: RiskManager, sample_portfolio: PortfolioState):
        size = risk.calculate_position_size(sample_portfolio)
        # default_position_pct = 5% of 10000 = 500
        assert size == pytest.approx(500.0, abs=1)

    def test_kelly_sizing_over_50_trades(self, risk: RiskManager, experienced_portfolio: PortfolioState):
        size = risk.calculate_position_size(experienced_portfolio)
        assert size > 0
        # Should not exceed MAX_POSITION_PCT (10%) of 10000 = 1000
        assert size <= 1000.0

    def test_max_position_cap(self, risk: RiskManager):
        port = PortfolioState(
            capital=10000, total_trades=60,
            win_rate=0.9, avg_win=20, avg_loss=1,
            consecutive_losses=0,
        )
        size = risk.calculate_position_size(port)
        assert size <= 1000.0  # 10% cap


class TestAntiMartingale:
    def test_no_reduction_zero_losses(self, risk: RiskManager):
        assert risk.apply_anti_martingale(500.0, 0) == 500.0

    def test_reduction_on_losses(self, risk: RiskManager):
        # 3 losses: factor = max(0.30, 1.0 - 0.45) = 0.55
        result = risk.apply_anti_martingale(500.0, 3)
        assert result == pytest.approx(275.0, abs=1)

    def test_minimum_factor(self, risk: RiskManager):
        # 4 losses: factor = max(0.30, 1.0 - 0.60) = 0.40
        result = risk.apply_anti_martingale(500.0, 4)
        assert result == pytest.approx(200.0, abs=1)

    def test_pause_at_5_losses(self, risk: RiskManager):
        assert risk.apply_anti_martingale(500.0, 5) == 0.0

    def test_pause_at_10_losses(self, risk: RiskManager):
        assert risk.apply_anti_martingale(500.0, 10) == 0.0


class TestLiquidityFilter:
    def test_passes_normal_orderbook(self, risk: RiskManager, sample_orderbook: OrderBook):
        # position_size must be small enough: slippage = size/depth*100 < 3%
        # depth_5_usd=1150, so max size for 3% slippage = 1150*0.03 = 34.5
        passes, reason = risk.check_liquidity(sample_orderbook, position_size_usd=30)
        assert passes is True
        assert reason == ""

    def test_rejects_wide_spread(self, risk: RiskManager, thin_orderbook: OrderBook):
        passes, reason = risk.check_liquidity(thin_orderbook, position_size_usd=100)
        assert passes is False
        assert "spread" in reason

    def test_rejects_insufficient_depth(self, risk: RiskManager):
        ob = OrderBook(
            spread_pct=2.0,
            depth_5_usd=100.0,  # Below MIN_DEPTH_USD (500)
        )
        passes, reason = risk.check_liquidity(ob, position_size_usd=50)
        assert passes is False
        assert "depth" in reason

    def test_rejects_high_slippage(self, risk: RiskManager):
        ob = OrderBook(
            spread_pct=2.0,
            depth_5_usd=600.0,
        )
        # Position is 50% of depth → slippage ~50% >> MAX_SLIPPAGE_PCT (3%)
        passes, reason = risk.check_liquidity(ob, position_size_usd=300)
        assert passes is False
        assert "slippage" in reason


class TestCategoryLimits:
    def test_within_limit(self, risk: RiskManager):
        port = PortfolioState(capital=10000, category_exposure={"crypto": 1000})
        # crypto limit = 30% of 10000 = 3000. 1000 + 500 = 1500 <= 3000
        assert risk.check_category_limit("crypto", port, 500) is True

    def test_exceeds_limit(self, risk: RiskManager):
        port = PortfolioState(capital=10000, category_exposure={"crypto": 2800})
        # 2800 + 500 = 3300 > 3000
        assert risk.check_category_limit("crypto", port, 500) is False

    def test_unknown_category_uses_other(self, risk: RiskManager):
        port = PortfolioState(capital=10000, category_exposure={})
        # "other" limit = 30% = 3000
        assert risk.check_category_limit("other", port, 2500) is True
