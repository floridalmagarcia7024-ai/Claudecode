"""Shadow mode — lightweight virtual bots (Module 22).

Up to 2 shadow bots run in parallel with variant parameters.
They observe the real data flow but do NOT execute orders.
They record what they would have done and compute simulated PnL.

Graduation: if shadow outperforms real bot for 14 days → propose activation
via Telegram.

Resource-efficient: reuses data already collected by the main bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from config import settings

logger = structlog.get_logger(__name__)

MAX_SHADOW_BOTS = 2
GRADUATION_DAYS = 14


@dataclass
class ShadowTrade:
    """A virtual trade recorded by a shadow bot."""

    market_id: str
    direction: str
    entry_price: float
    exit_price: float = 0.0
    size_usd: float = 0.0
    pnl_usd: float = 0.0
    strategy: str = ""
    opened_at: str = ""
    closed_at: str = ""
    is_open: bool = True


@dataclass
class ShadowBotState:
    """State of a single shadow bot."""

    bot_id: str
    params: dict[str, float]
    created_at: str = ""
    trades: list[ShadowTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    num_trades: int = 0
    wins: int = 0
    equity_curve: list[float] = field(default_factory=lambda: [0.0])

    @property
    def win_rate(self) -> float:
        return self.wins / self.num_trades if self.num_trades > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 3:
            return 0.0
        import statistics
        import math
        returns = [
            self.equity_curve[i] - self.equity_curve[i - 1]
            for i in range(1, len(self.equity_curve))
        ]
        if not returns:
            return 0.0
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns) if len(returns) >= 2 else 1.0
        if std_r == 0:
            return 0.0
        return mean_r / std_r * math.sqrt(365)


class ShadowBotManager:
    """Manages up to 2 shadow bots running in parallel.

    Shadow bots receive the same market data as the real bot but only
    simulate decisions — no API calls, no real orders.
    """

    def __init__(self) -> None:
        self._bots: dict[str, ShadowBotState] = {}
        self._start_time: float = time.time()

    @property
    def bots(self) -> dict[str, ShadowBotState]:
        return self._bots

    def create_shadow_bot(self, bot_id: str, params: dict[str, float]) -> bool:
        """Create a new shadow bot with variant parameters.

        Returns False if max shadow bots reached.
        """
        if len(self._bots) >= MAX_SHADOW_BOTS:
            logger.warning("shadow_max_reached", max=MAX_SHADOW_BOTS)
            return False

        if bot_id in self._bots:
            logger.warning("shadow_already_exists", bot_id=bot_id)
            return False

        self._bots[bot_id] = ShadowBotState(
            bot_id=bot_id,
            params=params,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("shadow_bot_created", bot_id=bot_id, params=params)
        return True

    def remove_shadow_bot(self, bot_id: str) -> bool:
        """Remove a shadow bot."""
        if bot_id not in self._bots:
            return False
        del self._bots[bot_id]
        logger.info("shadow_bot_removed", bot_id=bot_id)
        return True

    def evaluate_signal(
        self,
        bot_id: str,
        market_id: str,
        direction: str,
        price: float,
        size_usd: float,
        strategy: str,
        z_score: float = 0.0,
        sentiment: float = 0.0,
        ai_confidence: float = 0.0,
    ) -> bool:
        """Evaluate a trading signal with shadow bot's parameters.

        The shadow bot applies its own thresholds to decide if it would
        have taken the trade. Returns True if the shadow would trade.
        """
        bot = self._bots.get(bot_id)
        if not bot:
            return False

        params = bot.params

        # Apply shadow bot's thresholds
        zscore_thresh = params.get("zscore_threshold", settings.zscore_threshold)
        sent_shift = params.get("sentiment_shift", settings.sentiment_shift)
        ai_min = params.get("ai_confidence_min", settings.ai_confidence_min)

        if strategy == "mean_reversion" and abs(z_score) < zscore_thresh:
            return False
        if abs(sentiment) < sent_shift:
            return False
        if ai_confidence > 0 and ai_confidence < ai_min:
            return False

        # Record virtual trade
        trade = ShadowTrade(
            market_id=market_id,
            direction=direction,
            entry_price=price,
            size_usd=size_usd,
            strategy=strategy,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        bot.trades.append(trade)
        logger.debug(
            "shadow_trade_opened",
            bot_id=bot_id,
            market_id=market_id,
            direction=direction,
        )
        return True

    def close_shadow_trade(
        self, bot_id: str, market_id: str, exit_price: float
    ) -> ShadowTrade | None:
        """Close a virtual shadow trade and compute PnL."""
        bot = self._bots.get(bot_id)
        if not bot:
            return None

        for trade in bot.trades:
            if trade.market_id == market_id and trade.is_open:
                trade.exit_price = exit_price
                trade.is_open = False
                trade.closed_at = datetime.now(timezone.utc).isoformat()

                # Calculate PnL
                if trade.direction == "BUY":
                    trade.pnl_usd = (exit_price - trade.entry_price) / trade.entry_price * trade.size_usd
                else:
                    trade.pnl_usd = (trade.entry_price - exit_price) / trade.entry_price * trade.size_usd

                # Deduct simulated fees
                fee = trade.size_usd * settings.paper_fee_pct / 100
                trade.pnl_usd -= fee

                bot.total_pnl += trade.pnl_usd
                bot.num_trades += 1
                if trade.pnl_usd > 0:
                    bot.wins += 1
                bot.equity_curve.append(bot.total_pnl)

                logger.debug(
                    "shadow_trade_closed",
                    bot_id=bot_id,
                    market_id=market_id,
                    pnl=round(trade.pnl_usd, 2),
                )
                return trade

        return None

    def update_open_positions(self, market_prices: dict[str, float]) -> None:
        """Update unrealized PnL for all open shadow trades."""
        for bot in self._bots.values():
            for trade in bot.trades:
                if trade.is_open and trade.market_id in market_prices:
                    price = market_prices[trade.market_id]
                    if trade.direction == "BUY":
                        trade.pnl_usd = (
                            (price - trade.entry_price) / trade.entry_price * trade.size_usd
                        )
                    else:
                        trade.pnl_usd = (
                            (trade.entry_price - price) / trade.entry_price * trade.size_usd
                        )

    def check_graduation(self, bot_id: str, real_pnl: float) -> bool:
        """Check if a shadow bot qualifies for graduation.

        Graduation criteria: outperforms real bot for 14+ days.
        """
        bot = self._bots.get(bot_id)
        if not bot:
            return False

        # Must be at least 14 days old
        try:
            created = datetime.fromisoformat(bot.created_at)
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            if age_days < GRADUATION_DAYS:
                return False
        except (ValueError, TypeError):
            return False

        # Must outperform real bot
        return bot.total_pnl > real_pnl and bot.num_trades >= 5

    def get_performance(self, bot_id: str | None = None) -> list[dict]:
        """Get performance data for dashboard comparison."""
        results = []
        bots = [self._bots[bot_id]] if bot_id and bot_id in self._bots else self._bots.values()

        for bot in bots:
            open_trades = sum(1 for t in bot.trades if t.is_open)
            closed_trades = sum(1 for t in bot.trades if not t.is_open)
            results.append({
                "bot_id": bot.bot_id,
                "params": bot.params,
                "total_pnl": round(bot.total_pnl, 2),
                "num_trades": bot.num_trades,
                "open_trades": open_trades,
                "closed_trades": closed_trades,
                "win_rate": round(bot.win_rate, 3),
                "sharpe_ratio": round(bot.sharpe_ratio, 3),
                "created_at": bot.created_at,
                "equity_curve": bot.equity_curve[-50:],  # Last 50 points
            })
        return results
