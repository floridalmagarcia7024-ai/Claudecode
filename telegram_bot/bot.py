"""Telegram alert system with graduated severity levels (Module 14 + Module 19).

4 levels:
  CRITICAL  -> send immediately, retry 3x
  IMPORTANT -> send immediately, once
  INFO      -> accumulate, send hourly summary
  DEBUG     -> log only, never Telegram

Anti-spam: same alert not repeated within 15 minutes.

Phase 2 commands: /status /balance /posiciones /help
Phase 3 commands: /start /stop /trades /metricas /calibracion
                  /backtest /regimen /config /exportar
Phase 4 commands: /optimizacion /aprobar_params /rechazar_params /shadow
                  /stress /divergencias
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from config import settings

logger = structlog.get_logger(__name__)

ANTI_SPAM_WINDOW_S = 15 * 60  # 15 minutes
INFO_SUMMARY_INTERVAL_S = 3600  # 1 hour


class AlertLevel(str, enum.Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class Alert:
    """A single alert message."""

    level: AlertLevel
    message: str
    key: str = ""  # For dedup
    timestamp: float = 0.0


class TelegramAlertBot:
    """Sends graduated alerts via Telegram and handles basic commands.

    Alert levels:
      CRITICAL:  Circuit breaker, balance < min, API key invalid
      IMPORTANT: Stop loss, API degraded >5min, 3+ consecutive losses
      INFO:      Trade opened/closed, signal generated, regime changed
      DEBUG:     API calls, ticks — log only
    """

    def __init__(self) -> None:
        self._bot = None
        self._chat_id = settings.telegram_chat_id
        self._enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self._last_sent: dict[str, float] = {}  # key -> timestamp
        self._info_buffer: list[str] = []
        self._running = False
        self._get_status_callback = None
        self._get_balance_callback = None
        self._get_positions_callback = None

        if self._enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=settings.telegram_bot_token)
                logger.info("telegram_bot_initialized")
            except ImportError:
                logger.warning("telegram_not_installed", msg="pip install python-telegram-bot")
                self._enabled = False
            except Exception as exc:
                logger.error("telegram_init_failed", error=str(exc))
                self._enabled = False

    def set_callbacks(
        self,
        status_cb=None,
        balance_cb=None,
        positions_cb=None,
        trades_cb=None,
        metrics_cb=None,
        calibration_cb=None,
        backtest_cb=None,
        regime_cb=None,
        export_cb=None,
        start_cb=None,
        stop_cb=None,
        # Phase 4 callbacks
        optimizer_cb=None,
        approve_params_cb=None,
        reject_params_cb=None,
        shadow_cb=None,
        stress_cb=None,
        divergences_cb=None,
    ) -> None:
        """Set callback functions for command responses."""
        self._get_status_callback = status_cb
        self._get_balance_callback = balance_cb
        self._get_positions_callback = positions_cb
        self._get_trades_callback = trades_cb
        self._get_metrics_callback = metrics_cb
        self._get_calibration_callback = calibration_cb
        self._get_backtest_callback = backtest_cb
        self._get_regime_callback = regime_cb
        self._get_export_callback = export_cb
        self._start_callback = start_cb
        self._stop_callback = stop_cb
        # Phase 4
        self._optimizer_callback = optimizer_cb
        self._approve_params_callback = approve_params_cb
        self._reject_params_callback = reject_params_cb
        self._shadow_callback = shadow_cb
        self._stress_callback = stress_cb
        self._divergences_callback = divergences_cb

    async def send_alert(self, level: AlertLevel, message: str, key: str = "") -> None:
        """Send an alert respecting level rules and anti-spam.

        Args:
            level: Severity level.
            message: Alert text.
            key: Dedup key (same key won't repeat within 15 min).
        """
        alert_key = key or message[:50]

        # DEBUG: log only
        if level == AlertLevel.DEBUG:
            logger.debug("alert_debug", message=message)
            return

        # Anti-spam check
        if self._is_spam(alert_key):
            logger.debug("alert_suppressed_spam", key=alert_key)
            return

        # INFO: buffer for hourly summary
        if level == AlertLevel.INFO:
            self._info_buffer.append(message)
            logger.info("alert_info_buffered", message=message[:80])
            return

        # CRITICAL and IMPORTANT: send immediately
        if level == AlertLevel.CRITICAL:
            prefix = "🚨 CRITICAL"
            await self._send_with_retry(f"{prefix}: {message}", retries=3)
        elif level == AlertLevel.IMPORTANT:
            prefix = "⚠️ IMPORTANT"
            await self._send_message(f"{prefix}: {message}")

        self._last_sent[alert_key] = time.time()

    async def flush_info_summary(self) -> None:
        """Send accumulated INFO alerts as a summary."""
        if not self._info_buffer:
            return

        summary = "📊 Hourly Summary:\n" + "\n".join(
            f"• {msg}" for msg in self._info_buffer[-20:]  # Cap at 20 items
        )
        await self._send_message(summary)
        self._info_buffer.clear()

    async def run_info_flush_loop(self) -> None:
        """Periodically flush INFO buffer."""
        self._running = True
        while self._running:
            await asyncio.sleep(INFO_SUMMARY_INTERVAL_S)
            try:
                await self.flush_info_summary()
            except Exception as exc:
                logger.error("info_flush_error", error=str(exc))

    async def stop(self) -> None:
        self._running = False
        await self.flush_info_summary()

    async def handle_command(self, command: str) -> str:
        """Handle a Telegram command and return response text.

        Phase 3: Full command set (Module 19).
        """
        cmd = command.strip().lower().split()[0] if command.strip() else ""

        if cmd == "/start":
            if self._start_callback:
                try:
                    await self._start_callback()
                    return "▶️ Bot started"
                except Exception as exc:
                    return f"Error starting bot: {exc}"
            return "Start callback not configured"

        elif cmd == "/stop":
            if self._stop_callback:
                try:
                    await self._stop_callback()
                    return "⏹ Bot stopped"
                except Exception as exc:
                    return f"Error stopping bot: {exc}"
            return "Stop callback not configured"

        elif cmd == "/status":
            if self._get_status_callback:
                try:
                    status = await self._get_status_callback()
                    return self._format_status(status)
                except Exception as exc:
                    return f"Error getting status: {exc}"
            return "Status callback not configured"

        elif cmd == "/balance":
            if self._get_balance_callback:
                try:
                    balance = await self._get_balance_callback()
                    return f"💰 Balance: ${balance:.2f} USDC"
                except Exception as exc:
                    return f"Error getting balance: {exc}"
            return "Balance callback not configured"

        elif cmd in ("/posiciones", "/positions"):
            if self._get_positions_callback:
                try:
                    positions = await self._get_positions_callback()
                    if not positions:
                        return "No active positions"
                    lines = ["📈 Active Positions:"]
                    for p in positions:
                        mid = p.get("market_id", "?")[:20]
                        d = p.get("direction", "?")
                        sz = p.get("size_usd", 0)
                        ep = p.get("entry_price", 0)
                        ts = p.get("trailing_state", "watching")
                        lines.append(f"• {mid} {d} ${sz:.0f} @ {ep:.4f} [{ts}]")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error getting positions: {exc}"
            return "Positions callback not configured"

        elif cmd == "/trades":
            if self._get_trades_callback:
                try:
                    trades = await self._get_trades_callback()
                    if not trades:
                        return "No recent trades"
                    lines = ["📋 Recent Trades:"]
                    for t in trades[:10]:
                        pnl = t.get("pnl_usd", 0)
                        icon = "✅" if pnl > 0 else "❌"
                        lines.append(
                            f"{icon} {t.get('market_id', '?')[:20]} "
                            f"${pnl:.2f} ({t.get('strategy', '?')})"
                        )
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error getting trades: {exc}"
            return "Trades callback not configured"

        elif cmd in ("/metricas", "/metrics"):
            if self._get_metrics_callback:
                try:
                    m = await self._get_metrics_callback()
                    return (
                        f"📊 Metrics\n"
                        f"Total PnL: ${m.get('total_pnl', 0):.2f}\n"
                        f"Daily PnL: ${m.get('daily_pnl', 0):.2f}\n"
                        f"Win Rate: {m.get('win_rate', 0) * 100:.1f}%\n"
                        f"Sharpe: {m.get('sharpe_ratio', 0):.3f}\n"
                        f"Max DD: ${m.get('max_drawdown', 0):.2f}\n"
                        f"Total Trades: {m.get('total_trades', 0)}"
                    )
                except Exception as exc:
                    return f"Error getting metrics: {exc}"
            return "Metrics callback not configured"

        elif cmd in ("/calibracion", "/calibration"):
            if self._get_calibration_callback:
                try:
                    c = await self._get_calibration_callback()
                    detected = c.get("signals_detected", 0)
                    executed = c.get("executed", 0)
                    rejected = detected - executed
                    # Find major rejection cause
                    causes = {
                        "spread": c.get("rejected_spread", 0),
                        "z-score": c.get("rejected_zscore", 0),
                        "AI": c.get("rejected_ai", 0),
                        "liquidity": c.get("rejected_liquidity", 0),
                    }
                    top_cause = max(causes, key=causes.get) if causes else "none"
                    top_count = causes.get(top_cause, 0)
                    return (
                        f"🔧 Calibration\n"
                        f"Signals detected: {detected} | Executed: {executed} | "
                        f"Rejected: {rejected}\n"
                        f"Top rejection: {top_cause} ({top_count} of {rejected})"
                    )
                except Exception as exc:
                    return f"Error getting calibration: {exc}"
            return "Calibration callback not configured"

        elif cmd == "/backtest":
            if self._get_backtest_callback:
                try:
                    results = await self._get_backtest_callback()
                    if not results:
                        return "No backtest results available"
                    lines = ["🧪 Backtest Results:"]
                    for r in results:
                        sem = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(
                            r.get("semaphore", "RED"), "🔴"
                        )
                        lines.append(
                            f"{sem} {r.get('strategy', '?')}: "
                            f"Sharpe={r.get('sharpe_ratio', 0):.3f} "
                            f"WR={r.get('win_rate', 0) * 100:.0f}% "
                            f"Trades={r.get('total_trades', 0)}"
                        )
                        for f in r.get("failure_reasons", []):
                            lines.append(f"  ⚠ {f}")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error getting backtest: {exc}"
            return "Backtest callback not configured"

        elif cmd in ("/regimen", "/regime"):
            if self._get_regime_callback:
                try:
                    regimes = await self._get_regime_callback()
                    if not regimes:
                        return "No regime data"
                    lines = ["📈 Market Regimes:"]
                    for market_id, regime in regimes.items():
                        lines.append(f"• {market_id[:20]}: {regime}")
                    return "\n".join(lines[:15])  # Cap
                except Exception as exc:
                    return f"Error getting regimes: {exc}"
            return "Regime callback not configured"

        elif cmd == "/config":
            return (
                f"⚙️ Current Config\n"
                f"Z-Score: {settings.zscore_threshold}\n"
                f"Max Spread: {settings.max_spread_pct}%\n"
                f"Sentiment Shift: {settings.sentiment_shift}\n"
                f"AI Confidence: {settings.ai_confidence_min}\n"
                f"Position Size: {settings.default_position_pct}%\n"
                f"Trailing Stop: {settings.trailing_pct}%\n"
                f"Mode: {'Paper' if settings.paper_mode else 'REAL'}"
            )

        elif cmd in ("/exportar", "/export"):
            if self._get_export_callback:
                try:
                    csv_data = await self._get_export_callback()
                    if csv_data:
                        return f"📄 Export ({len(csv_data)} bytes). Use dashboard for full CSV download."
                    return "No trades to export"
                except Exception as exc:
                    return f"Error exporting: {exc}"
            return "Export callback not configured"

        # ── Phase 4 Commands ─────────────────────────────────
        elif cmd in ("/optimizacion", "/optimization"):
            if self._optimizer_callback:
                try:
                    data = await self._optimizer_callback()
                    status = data.get("status", "idle")
                    lines = [f"🔧 Optimizer: {status}"]
                    if data.get("last_run"):
                        lines.append(f"Last run: {data['last_run']}")
                    lines.append(f"History: {data.get('history_count', 0)} runs")
                    proposal = data.get("proposal")
                    if proposal:
                        lines.append(
                            f"\n📋 Pending Proposal:\n"
                            f"  Improvement: {proposal.get('improvement_pct', 0):.1f}%\n"
                            f"  Sharpe: {proposal.get('sharpe', 0):.3f}\n"
                            f"  Paper Sharpe: {proposal.get('paper_sharpe', 0):.3f}\n"
                            f"  Params: {proposal.get('proposed_params', {})}"
                        )
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error: {exc}"
            return "Optimizer not configured"

        elif cmd == "/aprobar_params":
            if self._approve_params_callback:
                try:
                    result = await self._approve_params_callback()
                    if result:
                        return f"✅ Parameters approved and applied:\n{result}"
                    return "⚠️ No parameters pending approval"
                except Exception as exc:
                    return f"Error: {exc}"
            return "Approve callback not configured"

        elif cmd == "/rechazar_params":
            if self._reject_params_callback:
                try:
                    result = await self._reject_params_callback()
                    if result:
                        return "❌ Parameters rejected. Keeping current settings."
                    return "⚠️ No parameters pending rejection"
                except Exception as exc:
                    return f"Error: {exc}"
            return "Reject callback not configured"

        elif cmd == "/shadow":
            if self._shadow_callback:
                try:
                    data = await self._shadow_callback()
                    if not data:
                        return "No shadow bots active"
                    lines = ["👻 Shadow Bots:"]
                    for bot in data:
                        lines.append(
                            f"\n• {bot['bot_id']}\n"
                            f"  PnL: ${bot.get('total_pnl', 0):.2f}\n"
                            f"  Trades: {bot.get('num_trades', 0)}\n"
                            f"  Win Rate: {bot.get('win_rate', 0)*100:.0f}%\n"
                            f"  Sharpe: {bot.get('sharpe_ratio', 0):.3f}"
                        )
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error: {exc}"
            return "Shadow not configured"

        elif cmd == "/stress":
            if self._stress_callback:
                try:
                    data = await self._stress_callback()
                    if not data:
                        return "No stress test results yet"
                    lines = [
                        f"🏋️ Stress Test ({data.get('timestamp', '?')[:10]})\n"
                        f"Capital: ${data.get('capital', 0):.0f}\n"
                        f"Worst case: {data.get('worst_case_pct', 0):.1f}%"
                    ]
                    for s in data.get("scenarios", []):
                        cb = "🔴" if s.get("triggers_circuit_breaker") else "🟢"
                        lines.append(
                            f"{cb} {s['scenario']}: ${s.get('simulated_pnl', 0):.0f} "
                            f"({s.get('pnl_pct', 0):.1f}%)"
                        )
                    for alert in data.get("alerts", []):
                        lines.append(f"⚠️ {alert}")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error: {exc}"
            return "Stress test not configured"

        elif cmd in ("/divergencias", "/divergences"):
            if self._divergences_callback:
                try:
                    signals = await self._divergences_callback()
                    if not signals:
                        return "No divergences detected"
                    lines = ["🔀 Cross-Market Divergences:"]
                    for s in signals[:10]:
                        lines.append(
                            f"\n• {s.get('polymarket_question', '?')[:40]}\n"
                            f"  Poly: {s.get('polymarket_prob', 0):.1%} vs "
                            f"{s.get('external_platform', '?')}: {s.get('external_prob', 0):.1%}\n"
                            f"  Gap: {s.get('abs_divergence', 0):.1%}"
                        )
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error: {exc}"
            return "Cross-market not configured"

        elif cmd == "/help":
            return (
                "🤖 Polymarket Bot Commands:\n\n"
                "▶️ /start — Start the bot\n"
                "⏹ /stop — Stop the bot\n"
                "📊 /status — Bot status & health\n"
                "💰 /balance — Current USDC balance\n"
                "📈 /posiciones — Active positions\n"
                "📋 /trades — Recent trade history\n"
                "📊 /metricas — Performance metrics\n"
                "🔧 /calibracion — Signal calibration stats\n"
                "🧪 /backtest — Backtesting results\n"
                "📈 /regimen — Market regime states\n"
                "⚙️ /config — Current configuration\n"
                "📄 /exportar — Export trades\n"
                "\n— Phase 4 —\n"
                "🔧 /optimizacion — Optimizer status\n"
                "✅ /aprobar_params — Approve proposed params\n"
                "❌ /rechazar_params — Reject proposed params\n"
                "👻 /shadow — Shadow bot performance\n"
                "🏋️ /stress — Stress test results\n"
                "🔀 /divergencias — Cross-market divergences\n"
                "❓ /help — Show this help"
            )

        return f"Unknown command: {command}. Type /help for available commands."

    def _is_spam(self, key: str) -> bool:
        """Check if same alert was sent within anti-spam window."""
        last = self._last_sent.get(key, 0)
        return (time.time() - last) < ANTI_SPAM_WINDOW_S

    async def _send_with_retry(self, text: str, retries: int = 3) -> bool:
        """Send a message with retry on failure."""
        for attempt in range(retries):
            success = await self._send_message(text)
            if success:
                return True
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
        logger.error("telegram_send_failed_all_retries", text=text[:80])
        return False

    async def _send_message(self, text: str) -> bool:
        """Send a single Telegram message."""
        if not self._enabled or not self._bot:
            logger.info("telegram_disabled", message=text[:80])
            return False

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text[:4096],  # Telegram limit
                parse_mode="HTML",
            )
            return True
        except Exception as exc:
            logger.error("telegram_send_error", error=str(exc))
            return False

    @staticmethod
    def _format_status(status: dict) -> str:
        """Format status dict as a human-readable message."""
        return (
            f"🤖 Bot Status\n"
            f"Status: {status.get('status', '?')}\n"
            f"Mode: {status.get('mode', '?')}\n"
            f"Degraded: {status.get('degraded', '?')}\n"
            f"Uptime: {status.get('uptime_s', 0)}s\n"
            f"Positions: {status.get('positions', 0)}\n"
            f"Daily PnL: ${status.get('daily_pnl', 0):.2f}\n"
            f"Markets: {status.get('markets_monitored', 0)}"
        )
