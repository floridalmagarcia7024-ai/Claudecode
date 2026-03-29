"""FastAPI entry point for the Polymarket Trading Bot.

Provides a /health endpoint and starts the trading engine as a background task.
Phase 2: integrates regime detector, news feed, health monitor, telegram, event calendar.
Phase 3: integrates dashboard, backtesting, smart exit, journal, full telegram commands.
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
from backtesting.backtester import Backtester
from backtesting.walk_forward import WalkForwardOptimizer
from config import settings
from core.engine import TradingEngine
from core.paper import PaperTradingEngine
from core.regime import MarketRegimeDetector
from core.risk import RiskManager
from core.state import PositionManager
from dashboard.main import dashboard_app, set_components
from execution.smart_exit import SmartExitManager
from intelligence.ai_analyzer import AIAnalyzer
from intelligence.event_calendar import CalendarEvent, EventCalendar
from intelligence.news_feed import NewsFeedPipeline
from journal.ai_journal import AIJournalAnalyzer
from journal.recorder import TradeJournalRecorder
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
        version="3.0.0",
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

    # Phase 3 components
    backtester = Backtester()
    walk_forward = WalkForwardOptimizer()
    smart_exit = SmartExitManager(client=client, state=state)
    journal = TradeJournalRecorder()
    await journal.initialize()
    ai_journal = AIJournalAnalyzer()

    # Strategies: mean reversion + momentum
    strategies = [
        MeanReversionStrategy(ai_analyzer=ai),
        MomentumStrategy(regime_detector=regime_detector),
    ]

    # Wire dashboard components
    set_components(
        engine=None,  # Set after engine creation
        state=state,
        collector=collector,
        backtester=backtester,
        walk_forward=walk_forward,
        smart_exit=smart_exit,
        journal=journal,
        regime_detector=regime_detector,
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

        # Attach Phase 3 components to engine
        engine._smart_exit = smart_exit  # type: ignore[attr-defined]
        engine._journal = journal  # type: ignore[attr-defined]
        engine._ai_journal = ai_journal  # type: ignore[attr-defined]
        engine._backtester = backtester  # type: ignore[attr-defined]

        # Update dashboard engine reference
        set_components(
            engine=engine,
            state=state,
            collector=collector,
            backtester=backtester,
            walk_forward=walk_forward,
            smart_exit=smart_exit,
            journal=journal,
            regime_detector=regime_detector,
        )

        # Phase 3: Set up full telegram callbacks (Module 19)
        async def _get_trades():
            trades = await state.get_trade_history(limit=10)
            return [
                {
                    "market_id": t.market_id,
                    "pnl_usd": t.pnl_usd,
                    "strategy": t.strategy,
                    "direction": t.direction,
                }
                for t in trades
            ]

        async def _get_metrics():
            import math
            import statistics as stat_mod

            trades = await state.get_trade_history(limit=10000)
            daily_pnl = await state.get_daily_pnl()
            stats = await state.get_trade_stats()
            pnls = [t.pnl_usd for t in trades]
            sharpe = 0.0
            if len(pnls) >= 2:
                mean_r = stat_mod.mean(pnls)
                std_r = stat_mod.stdev(pnls)
                if std_r > 0:
                    sharpe = mean_r / std_r * math.sqrt(365)
            equity = [0.0]
            for p in pnls:
                equity.append(equity[-1] + p)
            peak = max_dd = 0.0
            for v in equity:
                peak = max(peak, v)
                max_dd = max(max_dd, peak - v)
            return {
                "total_pnl": sum(pnls),
                "daily_pnl": daily_pnl.total_pnl,
                "win_rate": stats["win_rate"],
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "total_trades": len(trades),
            }

        async def _get_calibration():
            cal = getattr(engine, "_calibration_stats", None)
            return cal or {
                "signals_detected": 0, "rejected_spread": 0,
                "rejected_zscore": 0, "rejected_ai": 0,
                "rejected_liquidity": 0, "passed_filters": 0, "executed": 0,
            }

        async def _get_backtest():
            results = []
            for s in ["mean_reversion", "momentum"]:
                r = backtester.get_result(s)
                if r:
                    results.append(r.to_dict())
            return results

        async def _get_regimes():
            return dict(regime_detector._regimes)

        async def _get_positions_dict():
            positions = await state.get_active_positions()
            return [
                {
                    "market_id": p.market_id,
                    "direction": p.direction,
                    "size_usd": p.size_usd,
                    "entry_price": p.entry_price,
                    "trailing_state": p.trailing_state.value,
                }
                for p in positions
            ]

        telegram_bot.set_callbacks(
            status_cb=lambda: engine.get_status() if engine else {},
            balance_cb=lambda: client.get_balance(),
            positions_cb=_get_positions_dict,
            trades_cb=_get_trades,
            metrics_cb=_get_metrics,
            calibration_cb=_get_calibration,
            backtest_cb=_get_backtest,
            regime_cb=_get_regimes,
            export_cb=lambda: journal.export_csv(),
            start_cb=lambda: engine.start() if engine else None,
            stop_cb=lambda: engine.stop() if engine else None,
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
    await journal.close()
    await collector.close()
    await state.close()
    logger.info("bot_shutdown_complete")


# ── Main app mounts the dashboard ────────────────────────────────

app = FastAPI(title="Polymarket Trading Bot", version="3.0.0", lifespan=lifespan)

# Mount the Phase 3 dashboard app (includes all API endpoints + HTML)
app.mount("/", dashboard_app)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
