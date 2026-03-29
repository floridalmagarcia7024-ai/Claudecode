"""Trading engine — main orchestrator.

Runs concurrent async loops for market scanning, position management,
data collection, news feed, and health monitoring.

Phase 2 additions:
  - Market regime detection (Module 10)
  - News feed pipeline integration (Module 9)
  - Momentum strategy (Module 11)
  - Resolution imminence checks (Module 12A)
  - Event calendar checks (Module 12C)
  - Health monitor with degradation chain (Module 13)
  - Telegram alerts (Module 14)

Phase 4 additions:
  - Auto optimizer weekly run (Module 21)
  - Shadow bot signal forwarding (Module 22)
  - Stress testing daily (Module 23)
  - Cross-market divergence checks (Module 24)
  - Order flow monitor + manipulation detection (Module 25)
  - Liquidity profile recording (Module 25B)
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
from core.regime import MarketRegime, MarketRegimeDetector
from core.risk import RiskManager
from core.state import Position, PositionManager, TrailingState
from intelligence.event_calendar import EventCalendar
from intelligence.news_feed import MarketKeywords, NewsFeedPipeline, NewsSignal
from intelligence.resolution import ResolutionDetector
from monitoring.health import DegradationLevel, HealthMonitor
from strategies.base import BaseStrategy, MarketContext, Signal
from strategies.mean_reversion import MeanReversionStrategy
from telegram_bot.bot import AlertLevel, TelegramAlertBot

logger = structlog.get_logger(__name__)


class TradingEngine:
    """Main trading engine orchestrating all bot components.

    Runs concurrent async loops:
        1. Market scanning (every scan_interval_seconds)
        2. Position management (every position_check_interval_seconds)
        3. Data collection (delegated to DataCollector)
        4. News feed polling (every 60s) — Phase 2
        5. Health monitoring heartbeat (every 30s) — Phase 2
        6. Telegram info flush (every hour) — Phase 2
    """

    def __init__(
        self,
        client: PolymarketClient,
        state: PositionManager,
        risk: RiskManager,
        strategies: list[BaseStrategy],
        paper_engine: PaperTradingEngine,
        data_collector: DataCollector,
        regime_detector: MarketRegimeDetector | None = None,
        news_feed: NewsFeedPipeline | None = None,
        health_monitor: HealthMonitor | None = None,
        telegram_bot: TelegramAlertBot | None = None,
        event_calendar: EventCalendar | None = None,
        # Phase 4 components
        optimizer=None,
        shadow_manager=None,
        stress_tester=None,
        cross_market=None,
        order_flow=None,
        liquidity_profile=None,
    ) -> None:
        self._client = client
        self._state = state
        self._risk = risk
        self._strategies = strategies
        self._paper = paper_engine
        self._collector = data_collector
        self._regime = regime_detector or MarketRegimeDetector()
        self._news_feed = news_feed
        self._health = health_monitor
        self._telegram = telegram_bot
        self._event_calendar = event_calendar

        # Phase 4
        self._optimizer = optimizer
        self._shadow_manager = shadow_manager
        self._stress_tester = stress_tester
        self._cross_market = cross_market
        self._order_flow = order_flow
        self._liquidity_profile = liquidity_profile

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

    async def _get_balance(self) -> float:
        """Get current balance — simulated in paper mode, real otherwise."""
        if settings.paper_mode:
            # Paper mode: initial capital + realized PnL from closed trades
            daily_pnl = await self._state.get_daily_pnl()
            trades = await self._state.get_trade_history(limit=10000)
            total_realized = sum(t.pnl_usd for t in trades)
            open_positions = await self._state.get_active_positions()
            total_open = sum(p.size_usd for p in open_positions)
            return settings.paper_initial_balance + total_realized - total_open
        return await self._client.get_balance()

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

        # Phase 2: additional loops
        if self._news_feed:
            tasks.append(asyncio.create_task(self._news_feed.start(), name="news_feed"))
        if self._health:
            tasks.append(asyncio.create_task(self._health.run_heartbeat_loop(), name="health_monitor"))
        if self._telegram:
            tasks.append(asyncio.create_task(self._telegram.run_info_flush_loop(), name="telegram_flush"))

        # Phase 4: additional loops
        tasks.append(asyncio.create_task(self._phase4_periodic_loop(), name="phase4_periodic"))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("engine_stopped")
        except Exception as exc:
            logger.error("engine_fatal_error", error=str(exc))
            if self._telegram:
                await self._telegram.send_alert(
                    AlertLevel.CRITICAL, f"Engine fatal error: {exc}"
                )
            raise

    async def stop(self) -> None:
        """Signal all loops to stop."""
        self._running = False
        self._collector._running = False
        if self._news_feed:
            await self._news_feed.stop()
        if self._health:
            await self._health.stop()
        if self._telegram:
            await self._telegram.stop()
        logger.info("engine_stop_requested")

    # ── Market Scanning Loop ──────────────────────────────────

    async def _scan_markets_loop(self) -> None:
        """Periodically scan markets, generate signals, and execute trades."""
        while self._running:
            try:
                await self._process_cancellations()
                await self._scan_and_trade()
                # Process news feed signals
                if self._news_feed:
                    await self._process_news_signals()
            except Exception as exc:
                logger.error("scan_loop_error", error=str(exc))
            await asyncio.sleep(settings.scan_interval_seconds)

    async def _scan_and_trade(self) -> None:
        """Fetch markets, run strategies, process signals."""
        # Check health monitor degradation
        if self._health and not self._health.can_open_new:
            logger.warning(
                "scan_blocked_degradation",
                level=self._health.degradation_level.value,
            )
            return

        # Check API latency first
        latency = await self._client.check_latency()
        if latency > settings.api_latency_max_seconds:
            self._degraded = True
            if self._health:
                self._health.report_component_status("api", False)
            logger.warning(
                "degraded_mode",
                latency_s=round(latency, 2),
                max=settings.api_latency_max_seconds,
            )
            return
        self._degraded = False
        if self._health:
            self._health.report_component_status("api", True)

        # Get balance and check minimum capital
        balance = await self._get_balance()
        if balance < settings.min_capital_usd:
            logger.warning(
                "insufficient_capital",
                balance=balance,
                min_required=settings.min_capital_usd,
            )
            if self._telegram:
                await self._telegram.send_alert(
                    AlertLevel.CRITICAL,
                    f"Balance ${balance:.2f} below minimum ${settings.min_capital_usd}",
                    key="low_balance",
                )
            return

        # Check circuit breaker
        daily_pnl = await self._state.get_daily_pnl()
        if self._risk.check_circuit_breaker(daily_pnl.total_pnl, balance):
            if self._telegram:
                await self._telegram.send_alert(
                    AlertLevel.CRITICAL,
                    f"Circuit breaker triggered: PnL ${daily_pnl.total_pnl:.2f}",
                    key="circuit_breaker",
                )
            return  # Circuit breaker already logs

        # Fetch markets
        self._markets_cache = await self._client.get_active_markets()
        if not self._markets_cache:
            return

        # Update news feed market keywords
        if self._news_feed:
            self._update_news_feed_markets()

        # Build portfolio state
        portfolio = await self._state.get_portfolio_state(balance)

        # Anti-martingale pause check
        if portfolio.consecutive_losses >= 5:
            logger.warning(
                "trading_paused",
                reason="anti_martingale_5_consecutive_losses",
                consecutive_losses=portfolio.consecutive_losses,
            )
            if self._telegram and portfolio.consecutive_losses == 5:
                await self._telegram.send_alert(
                    AlertLevel.IMPORTANT,
                    f"Trading paused: {portfolio.consecutive_losses} consecutive losses",
                    key="anti_martingale",
                )
            return

        # Alert on 3+ consecutive losses
        if portfolio.consecutive_losses >= 3 and self._telegram:
            await self._telegram.send_alert(
                AlertLevel.IMPORTANT,
                f"{portfolio.consecutive_losses} consecutive losses",
                key="consecutive_losses",
            )

        # Evaluate each market with each strategy
        for market in self._markets_cache:
            # Skip if already have a position
            if await self._state.has_active_position(market.market_id):
                continue

            # Phase 2: Check resolution imminence (Module 12A)
            if self._risk.resolution_detector.should_block_new_positions(market.end_date):
                logger.debug("scan_blocked_resolution", market_id=market.market_id)
                continue

            # Phase 2: Check event calendar (Module 12C)
            if self._event_calendar:
                market_kws = NewsFeedPipeline.extract_keywords(market.question)
                if self._event_calendar.should_block_new_position(market_kws):
                    logger.debug("scan_blocked_event", market_id=market.market_id)
                    continue

            # Get historical data for z-score
            history = await self._collector.get_probability_series(
                market.market_id, days=7
            )

            # Phase 2: Detect regime (Module 10)
            if self._regime.should_check(market.market_id):
                self._regime.detect_regime(market.market_id, history)

            regime = self._regime.get_regime(market.market_id)

            for strategy in self._strategies:
                # Skip momentum in non-trending regimes (handled internally too)
                # Skip mean_reversion in trending regimes when regime is active
                if strategy.name == "mean_reversion" and regime in (
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN
                ):
                    continue
                if strategy.name == "momentum" and regime not in (
                    MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN
                ):
                    continue
                # HIGH_VOLATILITY: don't open new positions
                if regime == MarketRegime.HIGH_VOLATILITY:
                    continue

                context = MarketContext(
                    market=market,
                    probability_history=history,
                    portfolio=portfolio,
                )

                signal = await strategy.generate_signal(context)
                if signal is not None:
                    await self._process_signal(signal, portfolio, regime)

    async def _process_news_signals(self) -> None:
        """Process signals generated by the news feed pipeline."""
        if not self._news_feed:
            return

        signals = self._news_feed.drain_signals()
        if not signals:
            return

        balance = await self._get_balance()
        portfolio = await self._state.get_portfolio_state(balance)

        for news_signal in signals:
            if await self._state.has_active_position(news_signal.market_id):
                continue

            # Convert NewsSignal to a trading Signal
            direction = "BUY" if news_signal.direction == "UP" else "SELL"
            signal = Signal(
                market_id=news_signal.market_id,
                condition_id=news_signal.condition_id,
                token_id=news_signal.token_id,
                direction=direction,
                strength=news_signal.confidence,
                strategy="news_sentiment",
                category=news_signal.category,
                sentiment_score=news_signal.vader_score,
                ai_confidence=news_signal.confidence,
                metadata={"headline": news_signal.headline[:100], "source": news_signal.source},
            )

            regime = self._regime.get_regime(news_signal.market_id)
            await self._process_signal(signal, portfolio, regime)

    async def _process_signal(
        self, signal: Signal, portfolio: Any, regime: MarketRegime = MarketRegime.RANGING
    ) -> None:
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

        # Phase 2: Apply regime adjustment (HIGH_VOLATILITY = 50% reduction)
        size_usd = self._risk.apply_regime_adjustment(size_usd, regime.value)
        if size_usd <= 0:
            return

        # Phase 2: Apply resolution size multiplier (Module 12A)
        market_data = self._find_market(signal.market_id)
        if market_data:
            res_multiplier, block = self._risk.check_resolution_risk(market_data.end_date)
            if block:
                logger.info("signal_rejected_resolution", market_id=signal.market_id)
                return
            size_usd *= res_multiplier

        # Phase 2: Check correlation exposure (Module 12B)
        passes, reason = self._risk.check_correlation_risk(
            signal.market_id, size_usd, portfolio
        )
        if not passes:
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

        # Phase 4: Order flow confidence adjustment (Module 25A)
        if self._order_flow and orderbook:
            bids = [(b[0], b[1]) for b in getattr(orderbook, "bids", [])]
            asks = [(a[0], a[1]) for a in getattr(orderbook, "asks", [])]
            if bids or asks:
                self._order_flow.analyze_orderbook(signal.market_id, bids, asks)
            adj = self._order_flow.get_confidence_adjustment(
                signal.market_id, signal.direction
            )
            if adj != 0:
                logger.debug(
                    "order_flow_adjustment",
                    market_id=signal.market_id,
                    adjustment=adj,
                )

        # Phase 4: Liquidity profile priority (Module 25B)
        if self._liquidity_profile:
            priority = self._liquidity_profile.get_priority_multiplier(signal.market_id)
            size_usd *= priority

        # Phase 4: Forward signal to shadow bots (Module 22)
        price = orderbook.mid_price if orderbook and orderbook.mid_price > 0 else 0.5
        self._forward_signal_to_shadows(signal, price, size_usd)

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
            if self._telegram:
                await self._telegram.send_alert(
                    AlertLevel.INFO,
                    f"Trade opened: {signal.direction} ${size_usd:.0f} on {signal.market_id[:20]} ({signal.strategy})",
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

        # Phase 2: Check resolution imminence — close immediately if past due
        market_data = self._find_market(position.market_id)
        if market_data and self._risk.resolution_detector.should_close_immediately(market_data.end_date):
            alert_msg = self._risk.resolution_detector.check_position_alert(
                position.market_id, market_data.end_date
            )
            if alert_msg and self._telegram:
                await self._telegram.send_alert(AlertLevel.CRITICAL, alert_msg, key=f"resolution_{position.market_id}")
            await self._close_position(position, current_price, "resolution_imminent", orderbook)
            return

        # Phase 2: Event calendar — evaluate preventive close
        if self._event_calendar and market_data:
            market_kws = NewsFeedPipeline.extract_keywords(market_data.question)
            if self._event_calendar.should_evaluate_close(market_kws):
                # Close if in profit
                if position.direction == "BUY" and current_price > position.entry_price:
                    await self._close_position(position, current_price, "event_preventive_close", orderbook)
                    return
                elif position.direction == "SELL" and current_price < position.entry_price:
                    await self._close_position(position, current_price, "event_preventive_close", orderbook)
                    return

        # Update trailing stop
        new_state, new_sl = self._risk.update_trailing_stop(position, current_price)
        if new_state != position.trailing_state or new_sl != position.stop_loss:
            await self._state.update_trailing_stop(position.id, new_sl, new_state)  # type: ignore[arg-type]

        # Check if trailing stop triggers close
        if new_state == TrailingState.CLOSING:
            await self._close_position(position, current_price, "trailing_stop", orderbook)
            if self._telegram:
                await self._telegram.send_alert(
                    AlertLevel.IMPORTANT,
                    f"Stop loss triggered: {position.market_id[:20]} @ {current_price:.4f}",
                    key=f"stop_{position.id}",
                )
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

        # Phase 2: Check momentum exit
        elif position.strategy == "momentum":
            history = await self._collector.get_probability_series(
                position.market_id, days=1
            )
            if len(history) >= 12:
                change_1h = current_price - history[-12]
                from strategies.momentum import MomentumStrategy
                if MomentumStrategy.check_exit_conditions(change_1h):
                    await self._close_position(
                        position, current_price, "momentum_exit", orderbook
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

        if self._telegram:
            await self._telegram.send_alert(
                AlertLevel.INFO,
                f"Position closed: {position.market_id[:20]} reason={exit_reason} @ {current_price:.4f}",
            )

    # ── Phase 4: Periodic Loop ─────────────────────────────────

    async def _phase4_periodic_loop(self) -> None:
        """Run Phase 4 periodic tasks: optimizer, stress test, cross-market."""
        while self._running:
            try:
                # Stress testing — daily at 00:00 UTC (Module 23)
                if self._stress_tester and self._stress_tester.should_run():
                    await self._run_stress_test()

                # Auto optimizer — Sunday 02:00 UTC (Module 21)
                if self._optimizer and self._optimizer.should_run():
                    await self._run_optimizer()

                # Check if optimizer paper test is complete
                if self._optimizer and self._optimizer.check_paper_test_complete():
                    self._optimizer.promote_to_approval(0.0)
                    if self._telegram:
                        proposal = self._optimizer.current_proposal
                        if proposal:
                            await self._telegram.send_alert(
                                AlertLevel.IMPORTANT,
                                "Optimizer proposal ready for review. "
                                "Use /optimizacion to see details, "
                                "/aprobar_params or /rechazar_params to decide.",
                                key="optimizer_proposal",
                            )

                # Cross-market divergences (Module 24) — every hour
                if self._cross_market and self._markets_cache:
                    await self._check_cross_market()

                # Record liquidity profiles (Module 25B)
                if self._liquidity_profile and self._markets_cache:
                    for market in self._markets_cache[:20]:
                        if market.token_ids:
                            try:
                                ob = await self._client.get_orderbook(market.token_ids[0])
                                if ob and ob.spread_pct > 0:
                                    self._liquidity_profile.record_spread(
                                        market.market_id, ob.spread_pct
                                    )
                            except Exception:
                                pass

            except Exception as exc:
                logger.error("phase4_loop_error", error=str(exc))

            await asyncio.sleep(300)  # Every 5 minutes

    async def _run_stress_test(self) -> None:
        """Execute daily stress test."""
        positions = await self._state.get_active_positions()
        pos_dicts = [
            {
                "market_id": p.market_id,
                "direction": p.direction,
                "size_usd": p.size_usd,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
            }
            for p in positions
        ]
        balance = await self._get_balance()
        report = await self._stress_tester.run_stress_test(pos_dicts, balance)

        # Send alerts if needed
        if self._telegram:
            for alert_msg in report.alerts:
                await self._telegram.send_alert(
                    AlertLevel.CRITICAL, alert_msg, key="stress_alert"
                )
            if not report.alerts:
                await self._telegram.send_alert(
                    AlertLevel.INFO,
                    f"Stress test OK. Worst case: {report.worst_case_pct:.1f}%",
                )

    async def _run_optimizer(self) -> None:
        """Execute weekly parameter optimization."""
        try:
            # Gather snapshots for backtesting
            snapshots = []
            for market in self._markets_cache[:10]:
                history = await self._collector.get_history(
                    market.market_id, days=30
                )
                snapshots.extend(history)

            if snapshots:
                result = await self._optimizer.run_optimization(snapshots)
                self._optimizer.mark_run_complete()
                if result and self._telegram:
                    await self._telegram.send_alert(
                        AlertLevel.IMPORTANT,
                        f"Optimizer found {result.improvement_pct:.1f}% improvement. "
                        f"Starting 7-day paper test.",
                        key="optimizer_result",
                    )
        except Exception as exc:
            logger.error("optimizer_run_error", error=str(exc))

    async def _check_cross_market(self) -> None:
        """Check for cross-market divergences."""
        pm_markets = [
            {
                "market_id": m.market_id,
                "question": m.question,
                "probability": m.probability,
            }
            for m in self._markets_cache[:30]
        ]
        try:
            signals = await self._cross_market.check_divergences(pm_markets)
            if signals and self._telegram:
                for sig in signals[:3]:
                    await self._telegram.send_alert(
                        AlertLevel.INFO,
                        f"Divergence: {sig.polymarket_question[:40]} "
                        f"Poly={sig.polymarket_prob:.0%} vs "
                        f"{sig.external_platform}={sig.external_prob:.0%} "
                        f"(gap {sig.abs_divergence:.0%})",
                    )
        except Exception as exc:
            logger.warning("cross_market_error", error=str(exc))

    def _forward_signal_to_shadows(
        self, signal: Signal, price: float, size_usd: float
    ) -> None:
        """Forward a trading signal to all shadow bots for evaluation."""
        if not self._shadow_manager:
            return
        for bot_id in list(self._shadow_manager.bots.keys()):
            self._shadow_manager.evaluate_signal(
                bot_id=bot_id,
                market_id=signal.market_id,
                direction=signal.direction,
                price=price,
                size_usd=size_usd,
                strategy=signal.strategy,
                z_score=getattr(signal, "z_score", 0.0),
                sentiment=getattr(signal, "sentiment_score", 0.0),
                ai_confidence=getattr(signal, "ai_confidence", 0.0),
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

    # ── Helpers ───────────────────────────────────────────────

    def _find_market(self, market_id: str) -> MarketData | None:
        """Find a market in the current cache."""
        for m in self._markets_cache:
            if m.market_id == market_id:
                return m
        return None

    def _update_news_feed_markets(self) -> None:
        """Update news feed with current market keywords."""
        if not self._news_feed:
            return

        market_kws = []
        for m in self._markets_cache:
            keywords = NewsFeedPipeline.extract_keywords(m.question)
            if keywords:
                market_kws.append(
                    MarketKeywords(
                        market_id=m.market_id,
                        question=m.question,
                        keywords=keywords,
                        category=m.category,
                        probability=m.probability,
                        token_id=m.token_ids[0] if m.token_ids else "",
                        condition_id=m.condition_id,
                    )
                )
        self._news_feed.register_markets(market_kws)

    # ── Status ────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """Get current engine status for health check."""
        positions = await self._state.get_active_positions()
        daily_pnl = await self._state.get_daily_pnl()
        balance = await self._get_balance()
        now = datetime.now(timezone.utc).isoformat()

        status = {
            "status": "running" if self._running else "stopped",
            "mode": "paper" if settings.paper_mode else "real",
            "degraded": self._degraded,
            "uptime_s": round(self.uptime_seconds),
            "balance": round(balance, 2),
            "positions": len(positions),
            "daily_pnl": round(daily_pnl.total_pnl, 2),
            "daily_trades": daily_pnl.num_trades,
            "markets_monitored": len(self._markets_cache),
            "timestamp": now,
        }

        # Phase 2: add health info
        if self._health:
            status["health"] = self._health.get_status()

        return status
