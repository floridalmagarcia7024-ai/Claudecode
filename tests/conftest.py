"""Shared test fixtures for pytest."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from api.polymarket import MarketData, OrderBook, OrderBookLevel
from core.state import PositionManager, PortfolioState


@pytest_asyncio.fixture
async def position_manager(tmp_path):
    """Create a PositionManager with a temporary SQLite database."""
    db_path = str(tmp_path / "test_trades.db")
    pm = PositionManager(db_path=db_path)
    await pm.initialize()
    yield pm
    await pm.close()


@pytest.fixture
def sample_market() -> MarketData:
    """Sample market data for testing."""
    return MarketData(
        market_id="test-market-001",
        condition_id="0xabc123",
        question="Will Bitcoin exceed $100k by end of 2026?",
        category="crypto",
        token_ids=["token-yes-001", "token-no-001"],
        outcomes=["Yes", "No"],
        end_date="2026-12-31T00:00:00Z",
        active=True,
        volume_24h=50_000.0,
        probability=0.65,
    )


@pytest.fixture
def sample_orderbook() -> OrderBook:
    """Sample order book with realistic spread."""
    return OrderBook(
        token_id="token-yes-001",
        bids=[
            OrderBookLevel(price=0.63, size=500),
            OrderBookLevel(price=0.62, size=300),
            OrderBookLevel(price=0.61, size=200),
            OrderBookLevel(price=0.60, size=400),
            OrderBookLevel(price=0.59, size=600),
        ],
        asks=[
            OrderBookLevel(price=0.65, size=400),
            OrderBookLevel(price=0.66, size=300),
            OrderBookLevel(price=0.67, size=200),
            OrderBookLevel(price=0.68, size=350),
            OrderBookLevel(price=0.69, size=500),
        ],
        best_bid=0.63,
        best_ask=0.65,
        mid_price=0.64,
        spread_pct=3.125,  # (0.65-0.63)/0.64*100
        depth_5_usd=1150.0,  # Approximate
    )


@pytest.fixture
def thin_orderbook() -> OrderBook:
    """Thin order book that should fail liquidity checks."""
    return OrderBook(
        token_id="token-thin-001",
        bids=[OrderBookLevel(price=0.50, size=50)],
        asks=[OrderBookLevel(price=0.60, size=50)],
        best_bid=0.50,
        best_ask=0.60,
        mid_price=0.55,
        spread_pct=18.18,  # Very wide
        depth_5_usd=55.0,  # Very shallow
    )


@pytest.fixture
def sample_portfolio() -> PortfolioState:
    """Sample portfolio state for testing."""
    return PortfolioState(
        capital=10_000.0,
        active_positions=[],
        total_trades=0,
        win_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        consecutive_losses=0,
        daily_pnl=0.0,
        category_exposure={},
    )


@pytest.fixture
def experienced_portfolio() -> PortfolioState:
    """Portfolio with 50+ trades for Kelly criterion testing."""
    return PortfolioState(
        capital=10_000.0,
        active_positions=[],
        total_trades=60,
        win_rate=0.55,
        avg_win=8.0,
        avg_loss=5.0,
        consecutive_losses=0,
        daily_pnl=0.0,
        category_exposure={},
    )


@pytest.fixture
def probability_history_normal() -> list[float]:
    """7-day probability history with mean ~0.60 and some variance."""
    import random
    random.seed(42)
    base = 0.60
    return [base + random.gauss(0, 0.03) for _ in range(200)]


@pytest.fixture
def probability_history_deviated() -> list[float]:
    """History where current price deviates significantly (for z-score > threshold)."""
    import random
    random.seed(42)
    # Mean around 0.60, std ~0.03
    history = [0.60 + random.gauss(0, 0.03) for _ in range(200)]
    return history
