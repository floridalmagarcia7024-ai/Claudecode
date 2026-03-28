"""Telegram alert system with graduated severity levels (Module 14).

4 levels:
  CRITICAL  -> send immediately, retry 3x
  IMPORTANT -> send immediately, once
  INFO      -> accumulate, send hourly summary
  DEBUG     -> log only, never Telegram

Anti-spam: same alert not repeated within 15 minutes.
Commands: /status /balance /posiciones /help
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
    ) -> None:
        """Set callback functions for command responses."""
        self._get_status_callback = status_cb
        self._get_balance_callback = balance_cb
        self._get_positions_callback = positions_cb

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
        """Handle a Telegram command and return response text."""
        cmd = command.strip().lower()

        if cmd == "/status":
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
                        lines.append(
                            f"• {p.get('market_id', '?')[:20]} "
                            f"{p.get('direction', '?')} "
                            f"${p.get('size_usd', 0):.0f} "
                            f"@ {p.get('entry_price', 0):.4f}"
                        )
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Error getting positions: {exc}"
            return "Positions callback not configured"

        elif cmd == "/help":
            return (
                "🤖 Polymarket Bot Commands:\n"
                "/status — Bot status & health\n"
                "/balance — Current USDC balance\n"
                "/posiciones — Active positions\n"
                "/help — Show this help"
            )

        return f"Unknown command: {command}"

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
