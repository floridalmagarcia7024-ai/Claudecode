"""Async Polymarket CLOB client with rate limiting and retry.

Wraps the synchronous py-clob-client with asyncio.to_thread(),
adds token-bucket rate limiting and exponential backoff retry.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from py_clob_client.client import ClobClient

from config import settings

logger = structlog.get_logger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────


class TokenBucketRateLimiter:
    """Token bucket rate limiter with configurable buffer and warning threshold."""

    def __init__(
        self,
        max_rpm: int = settings.rate_limit_rpm,
        buffer_pct: float = settings.rate_limit_buffer_pct,
        warn_pct: float = settings.rate_limit_warn_pct,
    ) -> None:
        effective = int(max_rpm * (1 - buffer_pct))
        self._max_tokens = effective
        self._tokens = float(effective)
        self._refill_rate = effective / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._warn_threshold = int(effective * warn_pct)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        async with self._lock:
            self._refill()
            while self._tokens < 1:
                wait = (1 - self._tokens) / self._refill_rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1
            used = self._max_tokens - int(self._tokens)
            if used >= self._warn_threshold:
                logger.warning(
                    "rate_limit_high_usage",
                    used=used,
                    max=self._max_tokens,
                    pct=round(used / self._max_tokens * 100, 1),
                )

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


# ── Data Types ────────────────────────────────────────────────


@dataclass
class MarketData:
    """Represents a Polymarket market with its current state."""

    market_id: str = ""
    condition_id: str = ""
    question: str = ""
    category: str = "other"
    token_ids: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    end_date: str = ""
    active: bool = True
    volume_24h: float = 0.0
    probability: float = 0.0  # Current implied probability (first outcome)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    size: float


@dataclass
class OrderBook:
    """Order book snapshot for a token."""

    token_id: str = ""
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread_pct: float = 0.0
    depth_5_usd: float = 0.0


# ── Client ────────────────────────────────────────────────────


class PolymarketClient:
    """Async wrapper around py-clob-client with rate limiting and retry."""

    def __init__(self, clob_client: ClobClient) -> None:
        self._client = clob_client
        self._rate_limiter = TokenBucketRateLimiter()

    async def _call_with_retry(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a sync function in a thread with retry and rate limiting."""
        delays = settings.retry_delays
        last_exc: Exception | None = None

        for attempt in range(len(delays) + 1):
            await self._rate_limiter.acquire()
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=settings.request_timeout_seconds,
                )
                latency = time.monotonic() - start
                if latency > settings.api_latency_max_seconds:
                    logger.warning("api_high_latency", latency_s=round(latency, 2))
                return result
            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"Request timed out after {settings.request_timeout_seconds}s")
                logger.warning(
                    "api_timeout",
                    attempt=attempt + 1,
                    func=func.__name__,
                )
            except Exception as exc:
                last_exc = exc
                
                # ¡Esta es la condición clave para abortar rápido y no perder 40 segundos!
                error_str = str(exc)
                if "No orderbook exists" in error_str or "404" in error_str:
                    raise exc

                logger.warning(
                    "api_error",
                    attempt=attempt + 1,
                    func=func.__name__,
                    error=error_str,
                )

            if attempt < len(delays):
                delay = delays[attempt]
                logger.info("api_retry", delay_s=delay, attempt=attempt + 1)
                await asyncio.sleep(delay)

        logger.error("api_all_retries_exhausted", func=func.__name__, error=str(last_exc))
        raise last_exc  # type: ignore[misc]

    # ── Public Methods ────────────────────────────────────────

    async def get_active_markets(self, limit: int = 100) -> list[MarketData]:
        """Fetch active markets from Polymarket."""
        try:
            raw = await self._call_with_retry(self._client.get_markets, next_cursor="")
            markets: list[MarketData] = []
            data = raw if isinstance(raw, list) else raw.get("data", [])
            for item in data[:limit]:
                tokens = item.get("tokens", [])
                token_ids = [t.get("token_id", "") for t in tokens]
                outcomes = [t.get("outcome", "") for t in tokens]

                # Derive probability from first token price
                prob = 0.0
                if tokens:
                    prob = float(tokens[0].get("price", 0.0))

                category = self._categorize_market(
                    item.get("question", ""), item.get("category", "")
                )

                markets.append(
                    MarketData(
                        market_id=item.get("condition_id", ""),
                        condition_id=item.get("condition_id", ""),
                        question=item.get("question", ""),
                        category=category,
                        token_ids=token_ids,
                        outcomes=outcomes,
                        end_date=item.get("end_date_iso", ""),
                        active=item.get("active", True),
                        volume_24h=float(item.get("volume_num_24hr", 0.0)),
                        probability=prob,
                    )
                )
            logger.info("markets_fetched", count=len(markets))
            return markets
        except Exception as exc:
            logger.error("get_markets_failed", error=str(exc))
            return []

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch order book for a specific token."""
        try:
            raw = await self._call_with_retry(self._client.get_order_book, token_id)
        except Exception as exc:
            # Si no existe el orderbook, devolvemos uno vacío para no romper la estrategia
            logger.debug("orderbook_not_found", token_id=token_id, error=str(exc))
            return OrderBook(token_id=token_id)

        bids = [
            OrderBookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
            for b in raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
            for a in raw.get("asks", [])
        ]

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 1.0
        mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.0
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 100.0

        # Depth: sum of top 5 levels on each side
        bid_depth = sum(b.price * b.size for b in bids[:5])
        ask_depth = sum(a.price * a.size for a in asks[:5])
        depth_5_usd = bid_depth + ask_depth

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread_pct=spread_pct,
            depth_5_usd=depth_5_usd,
        )

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict[str, Any]:
        """Place a limit order on the CLOB.

        Args:
            token_id: The token to trade.
            side: "BUY" or "SELL".
            price: Limit price.
            size: Size in shares.

        Returns:
            Order response from the API.
        """
        from py_clob_client.order_builder.constants import BUY, SELL

        order_side = BUY if side.upper() == "BUY" else SELL
        order_args = {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": order_side,
        }
        signed_order = await asyncio.to_thread(
            self._client.create_and_post_order, order_args
        )
        logger.info(
            "order_placed",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
        )
        return signed_order  # type: ignore[return-value]

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an existing order."""
        result = await self._call_with_retry(self._client.cancel, order_id)
        logger.info("order_cancelled", order_id=order_id)
        return result  # type: ignore[return-value]

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions from the API."""
        try:
            result = await self._call_with_retry(self._client.get_positions)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("get_positions_failed", error=str(exc))
            return []

    async def get_balance(self) -> float:
        """Get current USDC balance."""
        try:
            result = await self._call_with_retry(self._client.get_balance_allowance)
            if isinstance(result, dict):
                return float(result.get("balance", 0.0))
            return 0.0
        except Exception as exc:
            logger.error("get_balance_failed", error=str(exc))
            return 0.0

    async def get_fee_rate_bps(self) -> int:
        """Get current fee rate in basis points."""
        try:
            result = await self._call_with_retry(self._client.get_tick_size)
            return int(result) if result else 0
        except Exception as exc:
            logger.warning("get_fee_rate_failed", error=str(exc))
            return 0

    async def check_latency(self) -> float:
        """Measure API round-trip latency in seconds."""
        start = time.monotonic()
        try:
            await self._call_with_retry(self._client.get_markets, next_cursor="")
        except Exception:
            pass
        return time.monotonic() - start

    @staticmethod
    def _categorize_market(question: str, category: str) -> str:
        """Categorize a market based on question text and category field."""
        text = f"{question} {category}".lower()
        if any(w in text for w in ("president", "election", "congress", "senate", "political", "politics", "vote")):
            return "politics"
        if any(w in text for w in ("bitcoin", "ethereum", "crypto", "btc", "eth", "token", "defi")):
            return "crypto"
        if any(w in text for w in ("nba", "nfl", "soccer", "football", "tennis", "sport", "match", "game", "championship")):
            return "sports"
        return "other"
