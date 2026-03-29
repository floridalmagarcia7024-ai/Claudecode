"""Trade journal recorder — automatic post-trade logging (Module 20).

Records the original signal context, sentiment, and related news
within a +/-2h window of the trade.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import aiosqlite
import structlog

from config import settings

logger = structlog.get_logger(__name__)

_JOURNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    position_id INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL DEFAULT '',
    entry_price REAL NOT NULL DEFAULT 0.0,
    exit_price REAL NOT NULL DEFAULT 0.0,
    pnl_usd REAL NOT NULL DEFAULT 0.0,
    size_usd REAL NOT NULL DEFAULT 0.0,
    signal_strength REAL NOT NULL DEFAULT 0.0,
    z_score REAL,
    sentiment_score REAL,
    ai_confidence REAL,
    regime TEXT NOT NULL DEFAULT '',
    exit_reason TEXT NOT NULL DEFAULT '',
    related_news TEXT NOT NULL DEFAULT '',
    ai_analysis TEXT NOT NULL DEFAULT '',
    opened_at TEXT NOT NULL DEFAULT '',
    closed_at TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_journal_market ON trade_journal(market_id);
CREATE INDEX IF NOT EXISTS idx_journal_closed ON trade_journal(closed_at);
"""


class TradeJournalRecorder:
    """Records detailed trade context for post-analysis."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.trades_db_path.replace(
            "trades.db", "journal.db"
        )
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_JOURNAL_SCHEMA)
        await self._db.commit()
        logger.info("trade_journal_initialized", db_path=self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("TradeJournalRecorder not initialized.")
        return self._db

    async def record_trade(
        self,
        trade_id: int,
        position_id: int,
        market_id: str,
        strategy: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        size_usd: float,
        signal_strength: float = 0.0,
        z_score: float | None = None,
        sentiment_score: float | None = None,
        ai_confidence: float | None = None,
        regime: str = "",
        exit_reason: str = "",
        related_news: str = "",
        ai_analysis: str = "",
        opened_at: str = "",
        closed_at: str = "",
    ) -> int:
        """Record a trade in the journal."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self.db.execute(
            """INSERT INTO trade_journal
               (trade_id, position_id, market_id, strategy, direction,
                entry_price, exit_price, pnl_usd, size_usd,
                signal_strength, z_score, sentiment_score, ai_confidence,
                regime, exit_reason, related_news, ai_analysis,
                opened_at, closed_at, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id, position_id, market_id, strategy, direction,
                entry_price, exit_price, pnl_usd, size_usd,
                signal_strength, z_score, sentiment_score, ai_confidence,
                regime, exit_reason, related_news, ai_analysis,
                opened_at, closed_at, now,
            ),
        )
        await self.db.commit()
        journal_id = cursor.lastrowid
        logger.info(
            "trade_journaled",
            journal_id=journal_id,
            trade_id=trade_id,
            market_id=market_id,
            pnl_usd=round(pnl_usd, 2),
        )
        return journal_id  # type: ignore[return-value]

    async def get_journal_entries(
        self, limit: int = 50, market_id: str | None = None
    ) -> list[dict]:
        """Retrieve journal entries."""
        if market_id:
            cursor = await self.db.execute(
                "SELECT * FROM trade_journal WHERE market_id=? ORDER BY closed_at DESC LIMIT ?",
                (market_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM trade_journal ORDER BY closed_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        columns = [
            "id", "trade_id", "position_id", "market_id", "strategy",
            "direction", "entry_price", "exit_price", "pnl_usd", "size_usd",
            "signal_strength", "z_score", "sentiment_score", "ai_confidence",
            "regime", "exit_reason", "related_news", "ai_analysis",
            "opened_at", "closed_at", "recorded_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    async def export_csv(self) -> str:
        """Export all journal entries as CSV string."""
        entries = await self.get_journal_entries(limit=10000)
        if not entries:
            return ""

        headers = "date,market_id,strategy,direction,entry_price,exit_price,pnl_usd,fees_usd,size_usd,exit_reason"
        lines = [headers]
        for e in entries:
            lines.append(
                f"{e.get('closed_at','')},{e.get('market_id','')},{e.get('strategy','')},"
                f"{e.get('direction','')},{e.get('entry_price',0)},{e.get('exit_price',0)},"
                f"{e.get('pnl_usd',0)},0,{e.get('size_usd',0)},{e.get('exit_reason','')}"
            )
        return "\n".join(lines)
