"""FastAPI entry point for the Polymarket Trading Bot.

Provides a /health endpoint and starts the trading engine as a background task.
Phase 2: integrates regime detector, news feed, health monitor, telegram, event calendar.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from api.auth import create_clob_client
from api.data_collector import DataCollector
from api.polymarket import PolymarketClient
from config import settings
from core.engine import TradingEngine
from core.paper import PaperTradingEngine
from core.regime import MarketRegimeDetector
from core.risk import RiskManager
from core.state import PositionManager
from intelligence.ai_analyzer import AIAnalyzer
from intelligence.event_calendar import CalendarEvent, EventCalendar
from intelligence.news_feed import NewsFeedPipeline
from monitoring.health import HealthMonitor
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from telegram_bot.bot import TelegramAlertBot

# ── Logging Setup ──────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.paper_mode else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_config().get("min_level", 0)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# ── Global References ──────────────────────────────────────────

engine: TradingEngine | None = None
engine_task: asyncio.Task | None = None  # type: ignore[type-arg]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown."""
    global engine, engine_task

    logger.info(
        "bot_starting",
        mode="paper" if settings.paper_mode else "real",
        version="2.0.0",
    )

    # Initialize components
    try:
        clob_client = create_clob_client()
    except Exception as exc:
        logger.error("auth_failed", error=str(exc))
        clob_client = None

    client = PolymarketClient(clob_client) if clob_client else None
    state = PositionManager()
    await state.initialize()

    collector = DataCollector(client) if client else DataCollector(None)  # type: ignore[arg-type]
    await collector.initialize()

    risk = RiskManager()
    ai = AIAnalyzer()
    paper = PaperTradingEngine(state)

    # Phase 2 components
    regime_detector = MarketRegimeDetector()
    news_feed = NewsFeedPipeline(ai_analyzer=ai, feeds=settings.rss_feeds)
    telegram_bot = TelegramAlertBot()
    event_calendar = EventCalendar()

    # Load configured events
    for evt_dict in settings.event_calendar:
        try:
            event_calendar.add_event(CalendarEvent(
                name=evt_dict.get("name", ""),
                timestamp=evt_dict.get("timestamp", ""),
                keywords=evt_dict.get("keywords", []),
                category=evt_dict.get("category", "other"),
                impact=evt_dict.get("impact", "medium"),
            ))
        except Exception:
            pass

    health_monitor = HealthMonitor(client, state) if client else None  # type: ignore[arg-type]

    # Strategies: mean reversion + momentum
    strategies = [
        MeanReversionStrategy(ai_analyzer=ai),
        MomentumStrategy(regime_detector=regime_detector),
    ]

    # Set up telegram callbacks
    if client:
        telegram_bot.set_callbacks(
            status_cb=lambda: engine.get_status() if engine else {},
            balance_cb=lambda: client.get_balance(),
            positions_cb=lambda: state.get_active_positions(),
        )

    if client:
        engine = TradingEngine(
            client=client,
            state=state,
            risk=risk,
            strategies=strategies,
            paper_engine=paper,
            data_collector=collector,
            regime_detector=regime_detector,
            news_feed=news_feed,
            health_monitor=health_monitor,
            telegram_bot=telegram_bot,
            event_calendar=event_calendar,
        )

        # Phase 2: Reconcile on restart (Module 13)
        if health_monitor and not settings.paper_mode:
            alerts = await health_monitor.reconcile_on_restart()
            for alert in alerts:
                logger.warning("reconcile_alert", msg=alert)

        engine_task = asyncio.create_task(engine.start())
        logger.info("engine_started")
    else:
        logger.warning("engine_not_started", reason="no_api_client")

    yield

    # Shutdown
    logger.info("bot_shutting_down")
    if engine:
        await engine.stop()
    if engine_task:
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
    await collector.close()
    await state.close()
    logger.info("bot_shutdown_complete")


app = FastAPI(title="Polymarket Trading Bot", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint for Railway monitoring."""
    if engine:
        return await engine.get_status()
    return {
        "status": "initializing",
        "mode": "paper" if settings.paper_mode else "real",
        "positions": 0,
        "uptime_s": 0,
    }


@app.get("/positions")
async def positions():
    """List active positions."""
    if engine:
        pos = await engine._state.get_active_positions()
        return [
            {
                "id": p.id,
                "market_id": p.market_id,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "size_usd": p.size_usd,
                "trailing_state": p.trailing_state.value,
                "strategy": p.strategy,
            }
            for p in pos
        ]
    return []


@app.get("/trades")
async def trades(limit: int = 20):
    """List recent trades."""
    if engine:
        trade_list = await engine._state.get_trade_history(limit=limit)
        return [
            {
                "id": t.id,
                "market_id": t.market_id,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl_usd": round(t.pnl_usd, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "exit_reason": t.exit_reason,
                "timestamp": t.timestamp,
            }
            for t in trade_list
        ]
    return []


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
