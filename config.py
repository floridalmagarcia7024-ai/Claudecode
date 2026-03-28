"""Central configuration for Polymarket Trading Bot.

All thresholds defined once here, referenced everywhere via `from config import settings`.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Bot configuration with environment variable overrides."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Trading Thresholds (Module 0) ──────────────────────────
    zscore_threshold: float = 1.8
    max_spread_pct: float = 6.0
    sentiment_shift: float = 0.25
    ai_confidence_min: float = 0.60
    min_daily_volume: float = 1_000.0
    correlation_block: float = 0.80
    max_daily_loss_pct: float = 8.0
    trailing_pct: float = 5.0
    breakeven_trigger: float = 5.0
    max_slippage_pct: float = 3.0
    min_depth_usd: float = 500.0
    max_position_pct: float = 10.0
    default_position_pct: float = 5.0
    paper_fee_pct: float = 1.25

    # ── Category Limits (hardcoded, not user-configurable) ─────
    category_limits: dict[str, float] = Field(
        default={
            "politics": 0.40,
            "crypto": 0.30,
            "sports": 0.20,
            "other": 0.30,
        },
    )

    # ── API Keys ───────────────────────────────────────────────
    polymarket_api_key: str = ""
    polymarket_secret: str = ""
    polymarket_private_key: str = ""
    wallet_address: str = ""
    groq_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Phase 2: RSS Feeds (Module 9) ────────────────────────
    rss_feeds: dict[str, list[str]] = Field(default={
        "politics": [
            "https://feeds.reuters.com/Reuters/PoliticsNews",
            "https://rss.politico.com/politico.xml",
            "https://thehill.com/rss/syndicator/19109",
        ],
        "crypto": [
            "https://www.coindesk.com/arc/outboundfeeds/rss",
            "https://cointelegraph.com/rss",
        ],
        "general": [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.reuters.com/reuters/topNews",
        ],
        "sports": [
            "https://www.espn.com/espn/rss/news",
        ],
    })

    # ── Phase 2: Event Calendar (Module 12C) ──────────────────
    event_calendar: list[dict] = Field(default=[])

    # ── Operational ────────────────────────────────────────────
    paper_mode: bool = True
    scan_interval_seconds: int = 60
    position_check_interval_seconds: int = 30
    data_collection_interval_seconds: int = 300
    groq_daily_limit: int = 300
    min_capital_usd: float = 50.0
    api_latency_max_seconds: float = 8.0
    data_retention_days: int = 90

    # ── Database ───────────────────────────────────────────────
    trades_db_path: str = "data/trades.db"
    market_history_db_path: str = "data/market_history.db"

    # ── Rate Limiting ──────────────────────────────────────────
    rate_limit_rpm: int = 60  # Conservative: ~60 effective after 20% buffer
    rate_limit_buffer_pct: float = 0.20
    rate_limit_warn_pct: float = 0.70

    # ── Retry ──────────────────────────────────────────────────
    retry_delays: list[float] = Field(default=[2.0, 8.0, 30.0])
    request_timeout_seconds: float = 10.0


settings = Settings()
