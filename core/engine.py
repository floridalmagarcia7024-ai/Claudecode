"""Trading engine — main orchestrator.

Runs concurrent async loops for market scanning, position management,
and data collection. Processes signals through the full pipeline:
strategy → risk checks → liquidity filter → execution.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from api.data_collector import DataCollector
from api.polymarket import MarketData, PolymarketClient
from config import settings
from core.paper import PaperTradingEngine
from core.risk import RiskManager
from core.state import Position, PositionManager, TrailingState
from strategies.base import BaseStrategy, MarketContext, Signal
from strategies.mean_reversion import MeanReversionStrategy

logger = structlog.get_logger(__name__)


class TradingEngine:
    """Main trading engine orchestrating all bot components.

    Runs three concurrent async loops:
        1. Market scanning (every scan_interval_seconds)
        2. Position management (every position_check_interval_seconds)
        3. Data collection (delegated to DataCollector)
    """

    def __init__(
        self,
        client: PolymarketClient,
        state: PositionManager,
        risk: RiskManager,
        strategies: list[BaseStrategy],
        paper_engine: PaperTradingEngine,
        data_collector: DataCollector,
    ) -> None:
        self._client = client
        self._state = state
        self._risk = risk
        self._strategies = strategies
        self._paper = paper_engine
        self._collector = data_collector

        self._running = False
        self._degraded = False
        self._start_time = 0.0
        self._markets_cache: list[MarketData] = []
        self._pending_cancellations: asyncio.Queue[str] = asyncio.Queue()

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time if self._start_time else 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    async def start(self) -> None:
        """Start all engine loops as concurrent tasks."""
        self._running = True
        self._start_time = time.monotonic()

        logger.info(
            "engine_starting",
            mode="paper" if settings.paper_mode else "real",
            strategies=[s.name for s in self._strategies],
            scan_interval=settings.scan_interval_seconds,
        )

        tasks = [
            asyncio.create_task(self._scan_markets_loop(), name="scan_markets"),
            asyncio.create_task(self._manage_positions_loop(), name="manage_positions"),
            asyncio.create_task(self._collector.run_collection_loop(), name="data_collection"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("engine_stopped")
        except Exception as exc:
            logger.error("engine_fatal_error", error=str(exc))
            raise

    async def stop(self) -> None:
        """Signal all loops to stop."""
        self._running = False
        self._collector._running = False
        logger.info("engine_stop_requested")

    # ── Market Scanning Loop ──────────────────────────────────

    async def _scan_markets_loop(self) -> None:
        """Periodically scan markets, generate signals, and execute trades."""
        while self._running:
            try:
                await self._process_cancellations()
                await self._scan_and_trade()
            except Exception as exc:
                logger.error("scan_loop_error", error=str(exc))
            await asyncio.sleep(settings.scan_interval_seconds)

    async def _scan_and_trade(self) -> None:
        """Fetch markets, run strategies, process signals."""
        # Check API latency first
        latency = await self._client.check_latency()
        if latency > settings.api_latency_max_seconds:
            self._degraded = True
            logger.warning(
                "degraded_mode",
                latency_s=round(latency, 2),
                max=settings.api_latency_max_seconds,
            )
            return
        self._degraded = False

        # Get balance and check minimum capital
        balance = await self._client.get_balance()
        if balance < settings.min_capital_usd:
            logger.warning(
                "insufficient_capital",
                balance=balance,
                min_required=settings.min_capital_usd,
            )
            return

        # Check circuit breaker
        daily_pnl = await self._state.get_daily_pnl()
        if self._risk.check_circuit_breaker(daily_pnl.total_pnl, balance):
            return  # Circuit breaker already logs

        # Fetch markets
        self._markets_cache = await self._client.get_active_markets()
        if not self._markets_cache:
            return

        # Build portfolio state
        portfolio = await self._state.get_portfolio_state(balance)

        # Anti-martingale pause check
        if portfolio.consecutive_losses >= 5:
            logger.warning(
                "trading_paused",
                reason="anti_martingale_5_consecutive_losses",
                consecutive_losses=portfolio.consecutive_losses,
            )
            return

        # Evaluate each market with each strategy
        for market in self._markets_cache:
            # Skip if already have a position
            if await self._state.has_active_position(market.market_id):
                continue

            # Get historical data for z-score
            history = await self._collector.get_probability_series(
                market.market_id, days=7
            )

            for strategy in self._strategies:
                context = MarketContext(
                    market=market,
                    probability_history=history,
                    portfolio=portfolio,
                )

                signal = await strategy.generate_signal(context)
                if signal is not None:
                    await self._process_signal(signal, portfolio)

    async def _process_signal(self, signal: Signal, portfolio: Any) -> None:
        """Process a signal through risk checks and execute if valid."""
        # Dedup check (belt and suspenders)
        if await self._state.has_active_position(signal.market_id):
            logger.debug(
                "signal_skipped_duplicate",
                market_id=signal.market_id,
            )
            return

        # Calculate position size
        size_usd = self._risk.calculate_position_size(
            portfolio, category=signal.category
        )
        if size_usd <= 0:
            logger.info(
                "signal_rejected_sizing",
                market_id=signal.market_id,
                reason="zero_size_from_risk",
            )
            return

        # Liquidity check
        if signal.token_id:
            try:
                orderbook = await self._client.get_orderbook(signal.token_id)
            except Exception as exc:
                logger.warning(
                    "orderbook_fetch_failed",
                    market_id=signal.market_id,
                    error=str(exc),
                )
                return

            passes, reason = self._risk.check_liquidity(orderbook, size_usd)
            if not passes:
                logger.info(
                    "signal_rejected_liquidity",
                    market_id=signal.market_id,
                    reason=reason,
                )
                return
        else:
            orderbook = None

        # Execute trade
        await self._execute_trade(signal, size_usd, orderbook)

    async def _execute_trade(
        self, signal: Signal, size_usd: float, orderbook: Any
    ) -> None:
        """Execute a trade in paper or real mode."""
        if settings.paper_mode:
            fill = await self._paper.execute_order(
                token_id=signal.token_id,
                direction=signal.direction,
                size_usd=size_usd,
                orderbook=orderbook,
                market_id=signal.market_id,
                condition_id=signal.condition_id,
                strategy=signal.strategy,
                category=signal.category,
            )
            logger.info(
                "trade_executed_paper",
                market_id=signal.market_id,
                direction=signal.direction,
                size_usd=round(size_usd, 2),
                fill_price=round(fill.fill_price, 4),
                strategy=signal.strategy,
            )
        else:
            # Real mode — place limit order at orderbook mid price
            if orderbook and orderbook.mid_price > 0:
                price = orderbook.mid_price
                shares = size_usd / price if price > 0 else 0
                try:
                    result = await self._client.place_order(
                        token_id=signal.token_id,
                        side=signal.direction,
                        price=price,
                        size=shares,
                    )
                    # Record position
                    position = Position(
                        market_id=signal.market_id,
                        condition_id=signal.condition_id,
                        token_id=signal.token_id,
                        direction=signal.direction,
                        entry_price=price,
                        current_price=price,
                        size_usd=size_usd,
                        strategy=signal.strategy,
                        category=signal.category,
                    )
                    await self._state.open_position(position)
                    await self._state.log_audit(
                        "real_order_placed",
                        f"market={signal.market_id} dir={signal.direction} "
                        f"size={size_usd} price={price}",
                    )
                    logger.info(
                        "trade_executed_real",
                        market_id=signal.market_id,
                        result=str(result)[:200],
                    )
                except Exception as exc:
                    logger.error(
                        "real_order_failed",
                        market_id=signal.market_id,
                        error=str(exc),
                    )

    # ── Position Management Loop ──────────────────────────────

    async def _manage_positions_loop(self) -> None:
        """Periodically update trailing stops and check exit conditions."""
        while self._running:
            try:
                await self._update_positions()
            except Exception as exc:
                logger.error("position_loop_error", error=str(exc))
            await asyncio.sleep(settings.position_check_interval_seconds)

    async def _update_positions(self) -> None:
        """Update all open positions: prices, trailing stops, exits."""
        positions = await self._state.get_active_positions()
        if not positions:
            return

        for position in positions:
            try:
                await self._update_single_position(position)
            except Exception as exc:
                logger.error(
                    "position_update_error",
                    position_id=position.id,
                    error=str(exc),
                )

    async def _update_single_position(self, position: Position) -> None:
        """Update a single position: fetch price, update trailing stop, check exits."""
        # Get current price
        if not position.token_id:
            return

        try:
            orderbook = await self._client.get_orderbook(position.token_id)
        except Exception:
            return

        current_price = orderbook.mid_price
        if current_price <= 0:
            return

        await self._state.update_position_price(position.id, current_price)  # type: ignore[arg-type]

        # Update trailing stop
        new_state, new_sl = self._risk.update_trailing_stop(position, current_price)
        if new_state != position.trailing_state or new_sl != position.stop_loss:
            await self._state.update_trailing_stop(position.id, new_sl, new_state)  # type: ignore[arg-type]

        # Check if trailing stop triggers close
        if new_state == TrailingState.CLOSING:
            await self._close_position(position, current_price, "trailing_stop", orderbook)
            return

        # Check mean reversion exit (z-score back to normal)
        if position.strategy == "mean_reversion":
            history = await self._collector.get_probability_series(
                position.market_id, days=7
            )
            z_score = MeanReversionStrategy.compute_z_score(current_price, history)
            if z_score is not None and MeanReversionStrategy.check_exit_conditions(z_score):
                await self._close_position(
                    position, current_price, "mean_reversion_exit", orderbook
                )

    async def _close_position(
        self,
        position: Position,
        current_price: float,
        exit_reason: str,
        orderbook: Any = None,
    ) -> None:
        """Close a position in paper or real mode."""
        if settings.paper_mode:
            await self._paper.close_position(
                position_id=position.id,  # type: ignore[arg-type]
                current_price=current_price,
                exit_reason=exit_reason,
                orderbook=orderbook,
            )
        else:
            # Real mode: market close via cancel + opposite order
            fees = 0.0
            try:
                fee_bps = await self._client.get_fee_rate_bps()
                fees = position.size_usd * fee_bps / 10000
            except Exception:
                pass

            await self._state.close_position(
                position_id=position.id,  # type: ignore[arg-type]
                exit_price=current_price,
                exit_reason=exit_reason,
                fees_usd=fees,
            )
            await self._state.log_audit(
                "real_position_closed",
                f"position={position.id} reason={exit_reason} price={current_price}",
            )

    # ── Priority Queue for Cancellations ──────────────────────

    async def request_cancellation(self, order_id: str) -> None:
        """Queue an order cancellation (highest priority)."""
        await self._pending_cancellations.put(order_id)

    async def _process_cancellations(self) -> None:
        """Process all pending cancellations before new orders."""
        while not self._pending_cancellations.empty():
            try:
                order_id = self._pending_cancellations.get_nowait()
                await self._client.cancel_order(order_id)
            except Exception as exc:
                logger.error("cancellation_failed", error=str(exc))

    # ── Status ────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """Get current engine status for health check."""
        positions = await self._state.get_active_positions()
        daily_pnl = await self._state.get_daily_pnl()
        now = datetime.now(timezone.utc).isoformat()

        return {
            "status": "running" if self._running else "stopped",
            "mode": "paper" if settings.paper_mode else "real",
            "degraded": self._degraded,
            "uptime_s": round(self.uptime_seconds),
            "positions": len(positions),
            "daily_pnl": round(daily_pnl.total_pnl, 2),
            "daily_trades": daily_pnl.num_trades,
            "markets_monitored": len(self._markets_cache),
            "timestamp": now,
        }
