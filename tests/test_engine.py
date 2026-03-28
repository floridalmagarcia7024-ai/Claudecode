"""Tests for TradingEngine and PositionManager."""

from __future__ import annotations

import pytest
import pytest_asyncio

from core.state import (
    DailyPnL,
    Position,
    PositionManager,
    PositionStatus,
    PortfolioState,
    TrailingState,
)


class TestPositionManager:
    @pytest.mark.asyncio
    async def test_open_and_get_position(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-001",
            condition_id="cond-001",
            token_id="tok-001",
            direction="BUY",
            entry_price=0.65,
            current_price=0.65,
            size_usd=500.0,
            strategy="mean_reversion",
            category="crypto",
        )
        pid = await position_manager.open_position(pos)
        assert pid > 0

        fetched = await position_manager.get_position_by_id(pid)
        assert fetched is not None
        assert fetched.market_id == "mkt-001"
        assert fetched.direction == "BUY"
        assert fetched.status == PositionStatus.OPEN

    @pytest.mark.asyncio
    async def test_has_active_position(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-unique",
            direction="BUY",
            entry_price=0.50,
            size_usd=100,
        )
        await position_manager.open_position(pos)
        assert await position_manager.has_active_position("mkt-unique") is True
        assert await position_manager.has_active_position("mkt-nonexistent") is False

    @pytest.mark.asyncio
    async def test_close_position_records_trade(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-close",
            direction="BUY",
            entry_price=0.50,
            current_price=0.50,
            size_usd=100,
        )
        pid = await position_manager.open_position(pos)

        trade = await position_manager.close_position(
            position_id=pid, exit_price=0.55, exit_reason="take_profit", fees_usd=1.25
        )
        assert trade is not None
        assert trade.pnl_usd > 0  # Profitable
        assert trade.exit_reason == "take_profit"

        # Position should be closed now
        assert await position_manager.has_active_position("mkt-close") is False

    @pytest.mark.asyncio
    async def test_close_position_loss(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-loss",
            direction="BUY",
            entry_price=0.60,
            current_price=0.60,
            size_usd=100,
        )
        pid = await position_manager.open_position(pos)

        trade = await position_manager.close_position(
            position_id=pid, exit_price=0.55, exit_reason="stop_loss"
        )
        assert trade is not None
        assert trade.pnl_usd < 0

    @pytest.mark.asyncio
    async def test_get_active_positions(self, position_manager: PositionManager):
        for i in range(3):
            pos = Position(
                market_id=f"mkt-active-{i}",
                direction="BUY",
                entry_price=0.50,
                size_usd=100,
            )
            await position_manager.open_position(pos)

        active = await position_manager.get_active_positions()
        assert len(active) >= 3

    @pytest.mark.asyncio
    async def test_consecutive_losses(self, position_manager: PositionManager):
        # Create 3 losing trades
        for i in range(3):
            pos = Position(
                market_id=f"mkt-closs-{i}",
                direction="BUY",
                entry_price=0.60,
                size_usd=100,
            )
            pid = await position_manager.open_position(pos)
            await position_manager.close_position(pid, exit_price=0.55, exit_reason="stop")

        losses = await position_manager.get_consecutive_losses()
        assert losses >= 3

    @pytest.mark.asyncio
    async def test_daily_pnl(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-pnl",
            direction="BUY",
            entry_price=0.50,
            size_usd=100,
        )
        pid = await position_manager.open_position(pos)
        await position_manager.close_position(pid, exit_price=0.55, exit_reason="tp")

        daily = await position_manager.get_daily_pnl()
        assert daily.num_trades >= 1
        assert daily.total_pnl != 0

    @pytest.mark.asyncio
    async def test_update_trailing_stop(self, position_manager: PositionManager):
        pos = Position(
            market_id="mkt-trail",
            direction="BUY",
            entry_price=0.50,
            size_usd=100,
        )
        pid = await position_manager.open_position(pos)

        await position_manager.update_trailing_stop(
            pid, stop_loss=0.48, trailing_state=TrailingState.BREAKEVEN
        )

        fetched = await position_manager.get_position_by_id(pid)
        assert fetched is not None
        assert fetched.trailing_state == TrailingState.BREAKEVEN
        assert fetched.stop_loss == 0.48

    @pytest.mark.asyncio
    async def test_portfolio_state(self, position_manager: PositionManager):
        portfolio = await position_manager.get_portfolio_state(capital=10000)
        assert portfolio.capital == 10000
        assert isinstance(portfolio.category_exposure, dict)

    @pytest.mark.asyncio
    async def test_audit_log(self, position_manager: PositionManager):
        await position_manager.log_audit("test_action", "test details")
        # Should not raise

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, position_manager: PositionManager):
        result = await position_manager.close_position(99999, 0.50, "test")
        assert result is None
