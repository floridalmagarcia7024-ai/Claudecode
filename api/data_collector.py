"""Automatic market data collector.

Periodically snapshots market probabilities into SQLite
for strategy consumption and future backtesting.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import aiosqlite
import structlog

from api.polymarket import PolymarketClient
from config import settings

logger = structlog.get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    probability REAL NOT NULL,
    volume_24h REAL NOT NULL DEFAULT 0.0,
    best_bid REAL NOT NULL DEFAULT 0.0,
    best_ask REAL NOT NULL DEFAULT 0.0,
    spread_pct REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_ts
    ON market_snapshots(market_id, timestamp);
"""


class DataCollector:
    """Collects market data snapshots at regular intervals."""

    def __init__(
        self,
        client: PolymarketClient,
        db_path: str | None = None,
    ) -> None:
        self._client = client
        self._db_path = db_path or settings.market_history_db_path
        self._db: aiosqlite.Connection | None = None
        self._running = False

    async def initialize(self) -> None:
        """Open the database and create tables."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        logger.info("data_collector_initialized", db_path=self._db_path)

    async def close(self) -> None:
        """Stop collection and close database."""
        self._running = False
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("DataCollector not initialized.")
        return self._db

    async def run_collection_loop(self) -> None:
        """Run the collection loop — call as an asyncio task."""
        self._running = True
        interval = settings.data_collection_interval_seconds
        logger.info("data_collection_started", interval_s=interval)

        while self._running:
            try:
                await self.collect_snapshot()
                await self.cleanup_old_data()
            except Exception as exc:
                logger.error("data_collection_error", error=str(exc))
            await asyncio.sleep(interval)

    async def collect_snapshot(self) -> int:
        """Fetch current market data and store snapshots. Returns count of snapshots saved."""
        markets = await self._client.get_active_markets()
        if not markets:
            logger.warning("no_markets_to_collect")
            return 0

        now = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[str, str, str, float, float, float, float, float]] = []

        for market in markets:
            # Optionally fetch order book for spread data
            best_bid = 0.0
            best_ask = 0.0
            spread_pct = 0.0

            if market.token_ids:
                try:
                    ob = await self._client.get_orderbook(market.token_ids[0])
                    best_bid = ob.best_bid
                    best_ask = ob.best_ask
                    spread_pct = ob.spread_pct
                except Exception:
                    pass  # Use defaults if order book unavailable

            rows.append((
                now,
                market.market_id,
                market.condition_id,
                market.probability,
                market.volume_24h,
                best_bid,
                best_ask,
                spread_pct,
            ))

        await self.db.executemany(
            """INSERT INTO market_snapshots
               (timestamp, market_id, condition_id, probability,
                volume_24h, best_bid, best_ask, spread_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self.db.commit()
        logger.info("snapshots_collected", count=len(rows))
        return len(rows)

    async def get_history(
        self, market_id: str, days: int = 7
    ) -> list[dict[str, float | str]]:
        """Retrieve historical snapshots for a market.

        Args:
            market_id: The market to query.
            days: Number of days of history to retrieve.

        Returns:
            List of snapshot dicts with timestamp, probability, volume_24h, spread_pct.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor = await self.db.execute(
            """SELECT timestamp, probability, volume_24h, spread_pct
               FROM market_snapshots
               WHERE market_id=? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (market_id, cutoff),
        )
        rows = await cursor.fetchall()
        return [
            {
                "timestamp": r[0],
                "probability": r[1],
                "volume_24h": r[2],
                "spread_pct": r[3],
            }
            for r in rows
        ]

    async def get_probability_series(
        self, market_id: str, days: int = 7
    ) -> list[float]:
        """Get just the probability values for z-score calculation."""
        history = await self.get_history(market_id, days)
        return [h["probability"] for h in history]  # type: ignore[misc]

    async def cleanup_old_data(self) -> None:
        """Remove data older than the configured retention period."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=settings.data_retention_days)
        ).isoformat()
        cursor = await self.db.execute(
            "DELETE FROM market_snapshots WHERE timestamp < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        if deleted:
            await self.db.commit()
            logger.info("old_data_cleaned", deleted=deleted)
