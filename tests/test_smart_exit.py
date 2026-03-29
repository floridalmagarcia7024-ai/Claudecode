"""Tests for smart exit manager (Module 16)."""

from __future__ import annotations

import pytest

from api.polymarket import OrderBook, OrderBookLevel
from core.state import Position, TrailingState
from execution.smart_exit import SmartExitManager, ExitPlan, IMPACT_LOW, IMPACT_MED


# ── Fixtures ──────────────────────────────────────────────────

def _make_orderbook(
    best_bid: float = 0.50,
    best_ask: float = 0.52,
    depth_5_usd: float = 5000.0,
) -> OrderBook:
    return OrderBook(
        token_id="test_token",
        bids=[OrderBookLevel(price=best_bid, size=100)],
        asks=[OrderBookLevel(price=best_ask, size=100)],
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=(best_bid + best_ask) / 2,
        spread_pct=((best_ask - best_bid) / ((best_bid + best_ask) / 2)) * 100,
        depth_5_usd=depth_5_usd,
    )


def _make_position(
    direction: str = "BUY",
    entry_price: float = 0.50,
    size_usd: float = 100.0,
    market_id: str = "test_market",
) -> Position:
    return Position(
        id=1,
        market_id=market_id,
        token_id="test_token",
        direction=direction,
        entry_price=entry_price,
        current_price=entry_price,
        size_usd=size_usd,
        strategy="mean_reversion",
    )


# ── Impact Estimation Tests ──────────────────────────────────


class TestImpactEstimation:
    def test_low_impact(self):
        sm = SmartExitManager()
        impact = sm.estimate_market_impact(50.0, 5000.0)
        assert impact == 1.0
        assert impact <= IMPACT_LOW

    def test_medium_impact(self):
        sm = SmartExitManager()
        impact = sm.estimate_market_impact(100.0, 5000.0)
        assert impact == 2.0
        assert IMPACT_LOW < impact <= IMPACT_MED

    def test_high_impact(self):
        sm = SmartExitManager()
        impact = sm.estimate_market_impact(200.0, 5000.0)
        assert impact == 4.0
        assert impact > IMPACT_MED

    def test_zero_depth(self):
        sm = SmartExitManager()
        impact = sm.estimate_market_impact(100.0, 0.0)
        assert impact == 100.0


# ── Exit Plan Tests ──────────────────────────────────────────


class TestExitPlan:
    def test_immediate_exit(self):
        sm = SmartExitManager()
        pos = _make_position(size_usd=50)
        ob = _make_orderbook(depth_5_usd=5000)
        plan = sm.plan_exit(pos, ob)

        assert plan.method == "immediate"
        assert plan.num_parts == 1

    def test_twap_3_exit(self):
        sm = SmartExitManager()
        pos = _make_position(size_usd=100)
        ob = _make_orderbook(depth_5_usd=5000)
        plan = sm.plan_exit(pos, ob)

        assert plan.method == "twap_3"
        assert plan.num_parts == 3
        assert plan.interval_s == 60

    def test_twap_5_exit(self):
        sm = SmartExitManager()
        pos = _make_position(size_usd=200)
        ob = _make_orderbook(depth_5_usd=5000)
        plan = sm.plan_exit(pos, ob)

        assert plan.method == "twap_5"
        assert plan.num_parts == 5
        assert plan.interval_s == 120

    def test_high_urgency_always_immediate(self):
        sm = SmartExitManager()
        pos = _make_position(size_usd=200)
        ob = _make_orderbook(depth_5_usd=5000)
        plan = sm.plan_exit(pos, ob, urgency="high")

        assert plan.method == "immediate"
        assert plan.urgency == "high"


# ── Exit Price Estimation ────────────────────────────────────


class TestExitPriceEstimation:
    def test_buy_exit_price(self):
        sm = SmartExitManager()
        pos = _make_position(direction="BUY")
        ob = _make_orderbook()
        price = sm.estimate_exit_price(pos, ob)
        # Selling should have price <= mid
        assert price <= ob.mid_price

    def test_sell_exit_price(self):
        sm = SmartExitManager()
        pos = _make_position(direction="SELL")
        ob = _make_orderbook()
        price = sm.estimate_exit_price(pos, ob)
        # Buying back should have price >= mid
        assert price >= ob.mid_price

    def test_zero_mid_price(self):
        sm = SmartExitManager()
        pos = _make_position()
        ob = OrderBook(
            token_id="test_token", bids=[], asks=[],
            best_bid=0, best_ask=0, mid_price=0,
            spread_pct=0, depth_5_usd=0,
        )
        price = sm.estimate_exit_price(pos, ob)
        assert price == 0.0


# ── Liquidation Price Estimator ──────────────────────────────


class TestLiquidationEstimator:
    def test_reject_unprofitable_buy(self):
        sm = SmartExitManager()
        # Entry at 0.50, exit estimated at 0.49 with 2.5% fees -> unprofitable
        reject = sm.should_reject_trade(0.50, 0.49, "BUY", fees_pct=1.25)
        assert reject is True

    def test_accept_profitable_buy(self):
        sm = SmartExitManager()
        reject = sm.should_reject_trade(0.50, 0.55, "BUY", fees_pct=1.25)
        assert reject is False

    def test_reject_unprofitable_sell(self):
        sm = SmartExitManager()
        reject = sm.should_reject_trade(0.50, 0.52, "SELL", fees_pct=1.25)
        assert reject is True

    def test_accept_profitable_sell(self):
        sm = SmartExitManager()
        reject = sm.should_reject_trade(0.50, 0.45, "SELL", fees_pct=1.25)
        assert reject is False


# ── Slippage Recording ───────────────────────────────────────


class TestSlippageTracking:
    def test_record_and_sample(self):
        sm = SmartExitManager()
        for _ in range(10):
            sm.record_slippage("market_1", 0.3, 100.0)

        # Should have historical data now
        assert len(sm._slippage_history["market_1"]) == 10

    def test_slippage_size_adjustment(self):
        sm = SmartExitManager()
        for _ in range(10):
            sm.record_slippage("market_1", 0.3, 100.0)

        # Large order should have higher slippage
        small_slip = sm._estimate_slippage("market_1", 50.0)
        large_slip = sm._estimate_slippage("market_1", 200.0)
        assert large_slip >= small_slip

    def test_default_slippage_no_history(self):
        sm = SmartExitManager()
        slip = sm._estimate_slippage("unknown_market", 100.0)
        assert slip > 0


# ── Paper Mode Smart Exit ────────────────────────────────────


class TestPaperSmartExit:
    @pytest.mark.asyncio
    async def test_paper_smart_exit(self):
        sm = SmartExitManager()
        pos = _make_position()
        ob = _make_orderbook()

        result = await sm.execute_smart_exit(pos, ob)
        assert "final_price" in result
        assert result["final_price"] > 0
        assert "method" in result

    @pytest.mark.asyncio
    async def test_paper_urgent_exit(self):
        sm = SmartExitManager()
        pos = _make_position()
        ob = _make_orderbook()

        result = await sm.execute_smart_exit(pos, ob, urgency="high")
        assert result["method"] == "market_order_urgent"
