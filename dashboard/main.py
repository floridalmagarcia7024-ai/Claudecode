```python
"""Dashboard FastAPI app with JWT auth, rate limiting, and CORS (Module 17).

Phase 5 additions: log ring buffer, /api/logs, /api/logs/stream,
/api/settings GET, /api/settings/credentials POST, /api/mode/toggle POST.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings

logger = structlog.get_logger(__name__)

# ── Log Ring Buffer ────────────────────────────────────────────
_log_buffer: deque = deque(maxlen=500)


def capture_log(entry: dict) -> None:
    """Append a structured log entry to the in-memory ring buffer."""
    _log_buffer.append({
        "time": str(entry.get("timestamp", ""))[:19].replace("T", " "),
        "level": str(entry.get("level", "info")).upper(),
        "msg": str(entry.get("event", "")),
    })


# ── JWT config ─────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer(auto_error=False)

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)


def create_dashboard_app() -> FastAPI:
    app = FastAPI(
        title="Polymarket Bot Dashboard",
        version="3.0.0",
        docs_url="/api/docs",
    )

    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return Response(
            content='{"error": "Rate limit exceeded"}',
            status_code=429,
            media_type="application/json",
        )

    allowed_origins = os.environ.get("CORS_ORIGINS", "").split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
    if not allowed_origins:
        allowed_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    return app


dashboard_app = create_dashboard_app()

# ── Shared state (set by main.py during startup) ────────────────

_engine = None
_state = None
_collector = None
_backtester = None
_walk_forward = None
_smart_exit = None
_journal = None
_regime_detector = None
_optimizer = None
_shadow_manager = None
_stress_tester = None
_cross_market = None
_order_flow = None
_liquidity_profile = None
_ab_manager = None
_audit_logger = None


def set_components(
    engine=None, state=None, collector=None, backtester=None,
    walk_forward=None, smart_exit=None, journal=None, regime_detector=None,
    optimizer=None, shadow_manager=None, stress_tester=None,
    cross_market=None, order_flow=None, liquidity_profile=None,
    ab_manager=None, audit_logger=None,
):
    global _engine, _state, _collector, _backtester, _walk_forward
    global _smart_exit, _journal, _regime_detector
    global _optimizer, _shadow_manager, _stress_tester, _cross_market
    global _order_flow, _liquidity_profile, _ab_manager, _audit_logger
    _engine = engine
    _state = state
    _collector = collector
    _backtester = backtester
    _walk_forward = walk_forward
    _smart_exit = smart_exit
    _journal = journal
    _regime_detector = regime_detector
    _optimizer = optimizer
    _shadow_manager = shadow_manager
    _stress_tester = stress_tester
    _cross_market = cross_market
    _order_flow = order_flow
    _liquidity_profile = liquidity_profile
    _ab_manager = ab_manager
    _audit_logger = audit_logger


# ── Pydantic Models ───────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class ConfigUpdate(BaseModel):
    key: str
    value: Any


class CredentialsUpdate(BaseModel):
    polymarket_api_key: str = ""
    polymarket_secret: str = ""
    polymarket_private_key: str = ""
    wallet_address: str = ""
    groq_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


# ── JWT Auth ──────────────────────────────────────────────────


def create_jwt_token(subject: str) -> str:
    payload = {
        "sub": subject,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return verify_jwt_token(credentials.credentials)


# ── Auth ──────────────────────────────────────────────────────


@dashboard_app.post("/api/auth/login")
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest):
    expected_user = os.environ.get("DASHBOARD_USER", "admin")
    expected_pass = os.environ.get("DASHBOARD_PASS", "polymarket")
    if body.username == expected_user and body.password == expected_pass:
        token = create_jwt_token(body.username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ── Health / Status ───────────────────────────────────────────


@dashboard_app.get("/health")
async def health():
    if _engine:
        return await _engine.get_status()
    return {"status": "initializing"}


@dashboard_app.get("/api/status")
@limiter.limit("30/minute")
async def api_status(request: Request, user: dict = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not running")
    status = await _engine.get_status()
    if _regime_detector:
        status["regime_detector"] = "active"
    return status


# ── Bot Control ───────────────────────────────────────────────


@dashboard_app.post("/api/start")
@limiter.limit("5/minute")
async def api_start(request: Request, user: dict = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    if _engine.is_running:
        return {"status": "already_running"}
    return {"status": "start_requested"}


@dashboard_app.post("/api/stop")
@limiter.limit("5/minute")
async def api_stop(request: Request, user: dict = Depends(get_current_user)):
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    if not _engine.is_running:
        return {"status": "already_stopped"}
    await _engine.stop()
    return {"status": "stopped"}


@dashboard_app.post("/api/mode/toggle")
@limiter.limit("5/minute")
async def api_mode_toggle(request: Request, user: dict = Depends(get_current_user)):
    """Toggle between paper and real trading mode."""
    settings.paper_mode = not settings.paper_mode
    mode = "paper" if settings.paper_mode else "real"
    logger.warning("mode_toggled", mode=mode, user=user.get("sub"))
    return {"paper_mode": settings.paper_mode, "mode": mode}


# ── Positions & Trades ────────────────────────────────────────


@dashboard_app.get("/api/positions")
@limiter.limit("30/minute")
async def api_positions(request: Request, user: dict = Depends(get_current_user)):
    if not _state:
        return []
    positions = await _state.get_active_positions()
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
            "category": p.category,
            "stop_loss": p.stop_loss,
            "opened_at": p.opened_at,
        }
        for p in positions
    ]


@dashboard_app.post("/api/positions/{position_id}/close")
@limiter.limit("10/minute")
async def api_close_position(
    position_id: int, request: Request, user: dict = Depends(get_current_user)
):
    if not _state:
        raise HTTPException(status_code=503, detail="State not available")
    position = await _state.get_position_by_id(position_id)
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")

    if _smart_exit and _engine and _engine._client and position.token_id:
        try:
            ob = await _engine._client.get_orderbook(position.token_id)
            exit_result = await _smart_exit.execute_smart_exit(position, ob)
            exit_price = exit_result.get("final_price", position.current_price)
        except Exception:
            exit_price = position.current_price
    else:
        exit_price = position.current_price

    trade = await _state.close_position(
        position_id=position_id,
        exit_price=exit_price,
        exit_reason="manual_close",
        fees_usd=position.size_usd * settings.paper_fee_pct / 100,
    )
    return {"status": "closed", "pnl_usd": round(trade.pnl_usd, 2) if trade else 0}


@dashboard_app.get("/api/trades")
@limiter.limit("30/minute")
async def api_trades(
    request: Request,
    limit: int = 50,
    strategy: str | None = None,
    user: dict = Depends(get_current_user),
):
    if not _state:
        return []
    trades = await _state.get_trade_history(limit=limit)
    result = []
    for t in trades:
        if strategy and t.strategy != strategy:
            continue
        result.append({
            "id": t.id,
            "market_id": t.market_id,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "pnl_usd": round(t.pnl_usd, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "fees_usd": round(t.fees_usd, 2),
            "strategy": t.strategy,
            "exit_reason": t.exit_reason,
            "timestamp": t.timestamp,
        })
    return result


# ── Metrics & Calibration ─────────────────────────────────────


@dashboard_app.get("/api/metrics")
@limiter.limit("30/minute")
async def api_metrics(request: Request, user: dict = Depends(get_current_user)):
    if not _state:
        raise HTTPException(status_code=503, detail="State not available")

    trades = await _state.get_trade_history(limit=10000)
    daily_pnl = await _state.get_daily_pnl()
    stats = await _state.get_trade_stats()

    total_pnl = sum(t.pnl_usd for t in trades)
    pnls = [t.pnl_usd for t in trades]

    import math
    import statistics as stat_mod
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
        "total_pnl": round(total_pnl, 2),
        "daily_pnl": round(daily_pnl.total_pnl, 2),
        "total_trades": len(trades),
        "daily_trades": daily_pnl.num_trades,
        "win_rate": round(stats["win_rate"], 3),
        "avg_win": round(stats["avg_win"], 3),
        "avg_loss": round(stats["avg_loss"], 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
    }


@dashboard_app.get("/api/metrics/calibration")
@limiter.limit("30/minute")
async def api_calibration(request: Request, user: dict = Depends(get_current_user)):
    if not _engine:
        return {"signals_detected": 0, "signals_executed": 0}
    cal = getattr(_engine, "_calibration_stats", None)
    if cal:
        return cal
    return {
        "signals_detected": 0, "rejected_spread": 0, "rejected_zscore": 0,
        "rejected_ai": 0, "rejected_liquidity": 0, "passed_filters": 0, "executed": 0,
    }


# ── Config & Settings ─────────────────────────────────────────


@dashboard_app.post("/api/config")
@limiter.limit("10/minute")
async def api_config_update(
    request: Request, body: ConfigUpdate, user: dict = Depends(get_current_user)
):
    allowed_keys = {
        "zscore_threshold", "max_spread_pct", "sentiment_shift",
        "ai_confidence_min", "min_daily_volume", "trailing_pct",
        "breakeven_trigger", "max_slippage_pct", "default_position_pct",
        "scan_interval_seconds", "paper_initial_balance", "paper_mode",
        "max_daily_loss_pct",
    }
    if body.key not in allowed_keys:
        raise HTTPException(status_code=400, detail=f"Key '{body.key}' not configurable")
    try:
        current = getattr(settings, body.key)
        setattr(settings, body.key, type(current)(body.value))
        return {"status": "updated", "key": body.key, "value": body.value}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@dashboard_app.get("/api/settings")
@limiter.limit("30/minute")
async def api_get_settings(request: Request, user: dict = Depends(get_current_user)):
    """Return current settings. Credentials are masked for security."""
    def mask(val: str) -> str:
        if not val:
            return ""
        return val[:4] + "****" if len(val) > 4 else "****"

    return {
        "paper_mode": settings.paper_mode,
        "paper_initial_balance": settings.paper_initial_balance,
        "default_position_pct": settings.default_position_pct,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "trailing_pct": settings.trailing_pct,
        "zscore_threshold": settings.zscore_threshold,
        "scan_interval_seconds": settings.scan_interval_seconds,
        # Credentials: show masked or empty
        "has_polymarket_key": bool(settings.polymarket_api_key),
        "has_groq_key": bool(settings.groq_api_key),
        "has_telegram": bool(settings.telegram_bot_token),
        "wallet_address_masked": mask(settings.wallet_address),
        "telegram_chat_id": settings.telegram_chat_id,
    }


@dashboard_app.post("/api/settings/credentials")
@limiter.limit("5/minute")
async def api_save_credentials(
    request: Request, body: CredentialsUpdate, user: dict = Depends(get_current_user)
):
    """Persist API credentials to .env file and apply to current session.
    Only non-empty fields are updated — existing values are preserved.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    # Read existing .env
    existing: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()

    # Map of env key → (new value, settings attribute)
    updates = {
        "POLYMARKET_API_KEY": (body.polymarket_api_key, "polymarket_api_key"),
        "POLYMARKET_SECRET": (body.polymarket_secret, "polymarket_secret"),
        "POLYMARKET_PRIVATE_KEY": (body.polymarket_private_key, "polymarket_private_key"),
        "WALLET_ADDRESS": (body.wallet_address, "wallet_address"),
        "GROQ_API_KEY": (body.groq_api_key, "groq_api_key"),
        "TELEGRAM_BOT_TOKEN": (body.telegram_bot_token, "telegram_bot_token"),
        "TELEGRAM_CHAT_ID": (body.telegram_chat_id, "telegram_chat_id"),
    }

    changed = []
    for env_key, (new_val, attr) in updates.items():
        if new_val:  # Only update if user provided a value
            existing[env_key] = new_val
            setattr(settings, attr, new_val)
            changed.append(env_key)

    # Write updated .env
    try:
        lines = [f"{k}={v}" for k, v in existing.items()]
        with open(env_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write .env: {exc}")

    return {
        "status": "saved",
        "updated_fields": changed,
        "note": "Restart required for API connection changes to take effect.",
    }


# ── Log Stream ────────────────────────────────────────────────


@dashboard_app.get("/api/logs")
@limiter.limit("60/minute")
async def api_logs(
    request: Request,
    limit: int = 100,
    user: dict = Depends(get_current_user),
):
    """Return recent log entries from the in-memory ring buffer."""
    entries = list(_log_buffer)
    return list(reversed(entries[-limit:]))


@dashboard_app.get("/api/logs/stream")
async def api_logs_stream(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """Server-Sent Events stream of live log entries."""
    if not credentials:
        return Response(status_code=401)
    try:
        verify_jwt_token(credentials.credentials)
    except HTTPException:
        return Response(status_code=401)

    async def event_generator():
        last_index = len(_log_buffer)
        while True:
            if await request.is_disconnected():
                break
            current = list(_log_buffer)
            if len(current) > last_index:
                for entry in current[last_index:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                last_index = len(current)
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Backtest & Regime ─────────────────────────────────────────


@dashboard_app.get("/api/backtest/{strategy}")
@limiter.limit("10/minute")
async def api_backtest(
    strategy: str, request: Request, user: dict = Depends(get_current_user)
):
    if not _backtester:
        raise HTTPException(status_code=503, detail="Backtester not available")
    result = _backtester.get_result(strategy)
    if result:
        return result.to_dict()
    if _collector:
        from api.data_collector import DataCollector
        try:
            cursor = await _collector.db.execute(
                "SELECT DISTINCT market_id FROM market_snapshots LIMIT 1"
            )
            row = await cursor.fetchone()
            if row:
                history = await _collector.get_history(row[0], days=30)
                if history:
                    bt_result = await _backtester.run_backtest(strategy, history)
                    return bt_result.to_dict()
        except Exception:
            pass
    return {"strategy": strategy, "status": "no_data"}


@dashboard_app.get("/api/regime/{market_id}")
@limiter.limit("30/minute")
async def api_regime(
    market_id: str, request: Request, user: dict = Depends(get_current_user)
):
    if not _regime_detector:
        return {"market_id": market_id, "regime": "unknown"}
    regime = _regime_detector.get_regime(market_id)
    return {"market_id": market_id, "regime": regime.value}


@dashboard_app.get("/api/export/trades")
@limiter.limit("5/minute")
async def api_export_trades(
    request: Request, format: str = "csv", user: dict = Depends(get_current_user)
):
    if not _journal:
        raise HTTPException(status_code=503, detail="Journal not available")
    csv_data = await _journal.export_csv()
    if not csv_data:
        return Response(content="No trades to export", media_type="text/plain")
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades_export.csv"},
    )


# ── Phase 4 Endpoints ─────────────────────────────────────────


@dashboard_app.post("/api/optimizer/approve")
@limiter.limit("5/minute")
async def api_optimizer_approve(request: Request, user: dict = Depends(get_current_user)):
    if not _optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not available")
    result = _optimizer.approve_params()
    if result:
        return {"status": "approved", "params": result}
    raise HTTPException(status_code=400, detail="No parameters pending approval")


@dashboard_app.post("/api/optimizer/reject")
@limiter.limit("5/minute")
async def api_optimizer_reject(request: Request, user: dict = Depends(get_current_user)):
    if not _optimizer:
        raise HTTPException(status_code=503, detail="Optimizer not available")
    result = _optimizer.reject_params()
    if result:
        return {"status": "rejected"}
    raise HTTPException(status_code=400, detail="No parameters pending rejection")


@dashboard_app.get("/api/optimizer/status")
@limiter.limit("30/minute")
async def api_optimizer_status(request: Request, user: dict = Depends(get_current_user)):
    if not _optimizer:
        return {"status": "not_configured"}
    return _optimizer.get_status_dict()


@dashboard_app.get("/api/shadow/performance")
@limiter.limit("30/minute")
async def api_shadow_performance(request: Request, user: dict = Depends(get_current_user)):
    if not _shadow_manager:
        return []
    return _shadow_manager.get_performance()


@dashboard_app.get("/api/stress/latest")
@limiter.limit("30/minute")
async def api_stress_latest(request: Request, user: dict = Depends(get_current_user)):
    if not _stress_tester or not _stress_tester.last_report:
        return {"status": "no_data"}
    return _stress_tester.last_report.to_dict()


@dashboard_app.get("/api/cross-market/divergences")
@limiter.limit("30/minute")
async def api_divergences(request: Request, user: dict = Depends(get_current_user)):
    if not _cross_market:
        return []
    return _cross_market.get_recent_signals()


@dashboard_app.get("/api/order-flow/whales")
@limiter.limit("30/minute")
async def api_whale_summary(request: Request, user: dict = Depends(get_current_user)):
    if not _order_flow:
        return {}
    return _order_flow.get_whale_summary()


@dashboard_app.get("/api/order-flow/manipulation")
@limiter.limit("30/minute")
async def api_manipulation_events(request: Request, user: dict = Depends(get_current_user)):
    if not _order_flow:
        return []
    return _order_flow.get_manipulation_events()


@dashboard_app.get("/api/liquidity/profile/{market_id}")
@limiter.limit("30/minute")
async def api_liquidity_profile(
    market_id: str, request: Request, user: dict = Depends(get_current_user)
):
    if not _liquidity_profile:
        return {"market_id": market_id, "status": "not_configured"}
    return _liquidity_profile.get_profile_dict(market_id)


@dashboard_app.get("/api/ab-test/status")
@limiter.limit("30/minute")
async def api_ab_status(request: Request, user: dict = Depends(get_current_user)):
    if not _ab_manager:
        return {"ab_test_active": False, "wallets": 0}
    return _ab_manager.get_status()


@dashboard_app.get("/api/audit/summary")
@limiter.limit("30/minute")
async def api_audit_summary(request: Request, user: dict = Depends(get_current_user)):
    if not _audit_logger:
        return {"total_entries": 0}
    return await _audit_logger.get_summary()


@dashboard_app.get("/api/audit/entries")
@limiter.limit("30/minute")
async def api_audit_entries(
    request: Request,
    action: str | None = None,
    market_id: str | None = None,
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    if not _audit_logger:
        return []
    return await _audit_logger.get_entries(action=action, market_id=market_id, limit=limit)


# ── HTML Pages ────────────────────────────────────────────────


@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@dashboard_app.get("/calibration", response_class=HTMLResponse)
async def dashboard_calibration(request: Request):
    return templates.TemplateResponse(request, "calibration.html")


@dashboard_app.get("/onboarding", response_class=HTMLResponse)
async def dashboard_onboarding(request: Request):
    return templates.TemplateResponse(request, "onboarding.html")

