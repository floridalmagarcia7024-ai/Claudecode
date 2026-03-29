"""Append-only audit log for real mode (Module — Audit).

Complete immutable log of all trading actions for regulatory compliance
and post-mortem analysis. Every action in real mode is recorded with
timestamp, action type, details, and optional metadata.

The audit log is append-only — entries are NEVER modified or deleted.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

AUDIT_DB_PATH = "data/audit.db"


class AuditLogger:
    """Append-only audit log backed by SQLite.

    Records every trading action in real mode with full context.
    Provides query capabilities for compliance and analysis.
    """

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create audit database and table."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                market_id TEXT,
                direction TEXT,
                size_usd REAL,
                price REAL,
                metadata TEXT,
                session_id TEXT
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_market ON audit_log(market_id)"
        )
        await self._db.commit()
        logger.info("audit_logger_initialized", db=self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def log(
        self,
        action: str,
        details: str,
        market_id: str = "",
        direction: str = "",
        size_usd: float = 0.0,
        price: float = 0.0,
        metadata: dict | None = None,
        session_id: str = "",
    ) -> int:
        """Append an audit entry.

        Args:
            action: Action type (e.g., "order_placed", "position_closed",
                    "params_changed", "circuit_breaker_triggered").
            details: Human-readable description.
            market_id: Related market (if applicable).
            direction: BUY/SELL (if applicable).
            size_usd: Trade size (if applicable).
            price: Execution price (if applicable).
            metadata: Additional JSON-serializable data.
            session_id: Bot session identifier.

        Returns:
            The audit entry ID.
        """
        if not self._db:
            logger.warning("audit_not_initialized")
            return -1

        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        cursor = await self._db.execute(
            """
            INSERT INTO audit_log
            (timestamp, action, details, market_id, direction, size_usd, price, metadata, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now, action, details, market_id, direction, size_usd, price, meta_json, session_id),
        )
        await self._db.commit()

        entry_id = cursor.lastrowid or 0
        logger.debug(
            "audit_logged",
            action=action,
            market_id=market_id[:20] if market_id else "",
            entry_id=entry_id,
        )
        return entry_id

    async def get_entries(
        self,
        action: str | None = None,
        market_id: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit log entries.

        Args:
            action: Filter by action type.
            market_id: Filter by market.
            since: ISO timestamp — return entries after this time.
            limit: Max entries to return.

        Returns:
            List of audit entry dicts.
        """
        if not self._db:
            return []

        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list = []

        if action:
            query += " AND action = ?"
            params.append(action)
        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        entries = []
        for row in rows:
            entry = dict(zip(columns, row))
            if entry.get("metadata"):
                try:
                    entry["metadata"] = json.loads(entry["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
            entries.append(entry)

        return entries

    async def get_count(self, action: str | None = None) -> int:
        """Get total count of audit entries."""
        if not self._db:
            return 0

        if action:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action = ?", (action,)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM audit_log")

        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_summary(self) -> dict:
        """Get audit log summary for dashboard."""
        if not self._db:
            return {"total_entries": 0}

        total = await self.get_count()

        # Count by action type
        cursor = await self._db.execute(
            "SELECT action, COUNT(*) as cnt FROM audit_log GROUP BY action ORDER BY cnt DESC LIMIT 20"
        )
        rows = await cursor.fetchall()
        by_action = {row[0]: row[1] for row in rows}

        # Latest entry
        cursor = await self._db.execute(
            "SELECT timestamp FROM audit_log ORDER BY id DESC LIMIT 1"
        )
        latest_row = await cursor.fetchone()

        return {
            "total_entries": total,
            "by_action": by_action,
            "latest_entry": latest_row[0] if latest_row else None,
        }
