"""Position and trade state management with SQLite persistence.

Handles all CRUD operations for positions, trades, daily P&L, and audit logs.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite
import structlog

from config import settings

logger = structlog.get_logger(__name__)


class PositionStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class TrailingState(str, enum.Enum):
    WATCHING = "watching"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"
    CLOSING = "closing"


@dataclass
class Position:
    """Represents an active or closed position."""

    id: int | None = None
    market_id: str = ""
    condition_id: str = ""
    token_id: str = ""
    direction: str = ""  # BUY or SELL
    entry_price: float = 0.0
    current_price: float = 0.0
    size_usd: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    stop_loss: float = 0.0
    trailing_state: TrailingState = TrailingState.WATCHING
    strategy: str = ""
    category: str = "other"
    opened_at: str = ""
    closed_at: str | None = None


@dataclass
class Trade:
    """Represents a completed trade."""

    id: int | None = None
    position_id: int = 0
    market_id: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    size_usd: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees_usd: float = 0.0
    strategy: str = ""
    exit_reason: str = ""
    timestamp: str = ""


@dataclass
class DailyPnL:
    """Daily profit and loss summary."""

    date: str = ""
    total_pnl: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0


@dataclass
class PortfolioState:
    """Snapshot of current portfolio for sizing decisions."""

    capital: float = 0.0
    active_positions: list[Position] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    category_exposure: dict[str, float] = field(default_factory=dict)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    condition_id TEXT NOT NULL DEFAULT '',
    token_id TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL DEFAULT 0.0,
    size_usd REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    stop_loss REAL NOT NULL DEFAULT 0.0,
    trailing_state TEXT NOT NULL DEFAULT 'watching',
    strategy TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'other',
    opened_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    fees_usd REAL NOT NULL DEFAULT 0.0,
    strategy TEXT NOT NULL DEFAULT '',
    exit_reason TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT PRIMARY KEY,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    num_trades INTEGER NOT NULL DEFAULT 0,
    num_wins INTEGER NOT NULL DEFAULT 0,
    num_losses INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
"""


class PositionManager:
    """Manages positions, trades, and P&L in SQLite."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.trades_db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database and create tables if needed."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        logger.info("state_initialized", db_path=self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("PositionManager not initialized. Call initialize() first.")
        return self._db

    # ── Positions ──────────────────────────────────────────────

    async def open_position(self, position: Position) -> int:
        """Insert a new open position. Returns the position ID."""
        now = datetime.now(timezone.utc).isoformat()
        position.opened_at = position.opened_at or now
        cursor = await self.db.execute(
            """INSERT INTO positions
               (market_id, condition_id, token_id, direction, entry_price,
                current_price, size_usd, status, stop_loss, trailing_state,
                strategy, category, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                position.market_id,
                position.condition_id,
                position.token_id,
                position.direction,
                position.entry_price,
                position.current_price,
                position.size_usd,
                position.status.value,
                position.stop_loss,
                position.trailing_state.value,
                position.strategy,
                position.category,
                position.opened_at,
            ),
        )
        await self.db.commit()
        position.id = cursor.lastrowid
        logger.info(
            "position_opened",
            position_id=position.id,
            market_id=position.market_id,
            direction=position.direction,
            size_usd=position.size_usd,
        )
        return position.id  # type: ignore[return-value]

    async def close_position(
        self, position_id: int, exit_price: float, exit_reason: str, fees_usd: float = 0.0
    ) -> Trade | None:
        """Close a position and record the trade. Returns the Trade."""
        row = await self._fetch_position_row(position_id)
        if row is None:
            logger.warning("close_position_not_found", position_id=position_id)
            return None

        pos = self._row_to_position(row)
        now = datetime.now(timezone.utc).isoformat()

        # Calculate P&L
        if pos.direction == "BUY":
            pnl_usd = (exit_price - pos.entry_price) * pos.size_usd / pos.entry_price
        else:
            pnl_usd = (pos.entry_price - exit_price) * pos.size_usd / pos.entry_price
        pnl_usd -= fees_usd
        pnl_pct = (pnl_usd / pos.size_usd) * 100 if pos.size_usd else 0.0

        # Update position
        await self.db.execute(
            "UPDATE positions SET status=?, closed_at=?, current_price=? WHERE id=?",
            (PositionStatus.CLOSED.value, now, exit_price, position_id),
        )

        # Record trade
        trade = Trade(
            position_id=position_id,
            market_id=pos.market_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            fees_usd=fees_usd,
            strategy=pos.strategy,
            exit_reason=exit_reason,
            timestamp=now,
        )
        cursor = await self.db.execute(
            """INSERT INTO trades
               (position_id, market_id, direction, entry_price, exit_price,
                size_usd, pnl_usd, pnl_pct, fees_usd, strategy, exit_reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.position_id,
                trade.market_id,
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.size_usd,
                trade.pnl_usd,
                trade.pnl_pct,
                trade.fees_usd,
                trade.strategy,
                trade.exit_reason,
                trade.timestamp,
            ),
        )
        trade.id = cursor.lastrowid

        # Update daily P&L
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        is_win = 1 if pnl_usd > 0 else 0
        is_loss = 1 if pnl_usd < 0 else 0
        await self.db.execute(
            """INSERT INTO daily_pnl (date, total_pnl, num_trades, num_wins, num_losses)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 total_pnl = total_pnl + excluded.total_pnl,
                 num_trades = num_trades + 1,
                 num_wins = num_wins + excluded.num_wins,
                 num_losses = num_losses + excluded.num_losses""",
            (today, pnl_usd, is_win, is_loss),
        )
        await self.db.commit()

        logger.info(
            "position_closed",
            position_id=position_id,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=exit_reason,
        )
        return trade

    async def update_position_price(self, position_id: int, current_price: float) -> None:
        """Update the current price of a position."""
        await self.db.execute(
            "UPDATE positions SET current_price=? WHERE id=?",
            (current_price, position_id),
        )
        await self.db.commit()

    async def update_trailing_stop(
        self, position_id: int, stop_loss: float, trailing_state: TrailingState
    ) -> None:
        """Update trailing stop state and level for a position."""
        await self.db.execute(
            "UPDATE positions SET stop_loss=?, trailing_state=? WHERE id=?",
            (stop_loss, trailing_state.value, position_id),
        )
        await self.db.commit()

    async def get_active_positions(self) -> list[Position]:
        """Return all currently open positions."""
        cursor = await self.db.execute(
            "SELECT * FROM positions WHERE status=?", (PositionStatus.OPEN.value,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_position(r) for r in rows]

    async def has_active_position(self, market_id: str) -> bool:
        """Check if there is already an open position for a market."""
        cursor = await self.db.execute(
            "SELECT 1 FROM positions WHERE market_id=? AND status=? LIMIT 1",
            (market_id, PositionStatus.OPEN.value),
        )
        return await cursor.fetchone() is not None

    async def get_position_by_id(self, position_id: int) -> Position | None:
        """Fetch a single position by ID."""
        row = await self._fetch_position_row(position_id)
        return self._row_to_position(row) if row else None

    # ── Trades & P&L ──────────────────────────────────────────

    async def get_trade_history(self, limit: int = 100) -> list[Trade]:
        """Return recent trades, most recent first."""
        cursor = await self.db.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_trade(r) for r in rows]

    async def get_daily_pnl(self, date: str | None = None) -> DailyPnL:
        """Return P&L summary for a given date (default: today)."""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cursor = await self.db.execute("SELECT * FROM daily_pnl WHERE date=?", (date,))
        row = await cursor.fetchone()
        if row is None:
            return DailyPnL(date=date)
        return DailyPnL(
            date=row[0], total_pnl=row[1], num_trades=row[2], num_wins=row[3], num_losses=row[4]
        )

    async def get_consecutive_losses(self) -> int:
        """Count consecutive losses from the most recent trades."""
        cursor = await self.db.execute(
            "SELECT pnl_usd FROM trades ORDER BY timestamp DESC LIMIT 50"
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            if row[0] < 0:
                count += 1
            else:
                break
        return count

    async def get_total_trade_count(self) -> int:
        """Return total number of completed trades."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM trades")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_trade_stats(self) -> dict[str, float]:
        """Return win rate, avg win, avg loss from trade history."""
        cursor = await self.db.execute(
            "SELECT pnl_pct FROM trades ORDER BY timestamp DESC"
        )
        rows = await cursor.fetchall()
        if not rows:
            return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}

        wins = [r[0] for r in rows if r[0] > 0]
        losses = [abs(r[0]) for r in rows if r[0] < 0]
        total = len(rows)

        return {
            "win_rate": len(wins) / total if total else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        }

    async def get_portfolio_state(self, capital: float) -> PortfolioState:
        """Build a full portfolio snapshot for sizing decisions."""
        positions = await self.get_active_positions()
        stats = await self.get_trade_stats()
        consecutive_losses = await self.get_consecutive_losses()
        daily = await self.get_daily_pnl()
        total_trades = await self.get_total_trade_count()

        category_exposure: dict[str, float] = {}
        for pos in positions:
            cat = pos.category
            category_exposure[cat] = category_exposure.get(cat, 0.0) + pos.size_usd

        return PortfolioState(
            capital=capital,
            active_positions=positions,
            total_trades=total_trades,
            win_rate=stats["win_rate"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
            consecutive_losses=consecutive_losses,
            daily_pnl=daily.total_pnl,
            category_exposure=category_exposure,
        )

    # ── Audit Log ─────────────────────────────────────────────

    async def log_audit(self, action: str, details: str = "") -> None:
        """Append an entry to the audit log (append-only)."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO audit_log (timestamp, action, details) VALUES (?, ?, ?)",
            (now, action, details),
        )
        await self.db.commit()

    # ── Internal ──────────────────────────────────────────────

    async def _fetch_position_row(self, position_id: int) -> tuple | None:  # type: ignore[type-arg]
        cursor = await self.db.execute("SELECT * FROM positions WHERE id=?", (position_id,))
        return await cursor.fetchone()  # type: ignore[return-value]

    @staticmethod
    def _row_to_position(row: tuple) -> Position:  # type: ignore[type-arg]
        return Position(
            id=row[0],
            market_id=row[1],
            condition_id=row[2],
            token_id=row[3],
            direction=row[4],
            entry_price=row[5],
            current_price=row[6],
            size_usd=row[7],
            status=PositionStatus(row[8]),
            stop_loss=row[9],
            trailing_state=TrailingState(row[10]),
            strategy=row[11],
            category=row[12],
            opened_at=row[13],
            closed_at=row[14],
        )

    @staticmethod
    def _row_to_trade(row: tuple) -> Trade:  # type: ignore[type-arg]
        return Trade(
            id=row[0],
            position_id=row[1],
            market_id=row[2],
            direction=row[3],
            entry_price=row[4],
            exit_price=row[5],
            size_usd=row[6],
            pnl_usd=row[7],
            pnl_pct=row[8],
            fees_usd=row[9],
            strategy=row[10],
            exit_reason=row[11],
            timestamp=row[12],
        )
