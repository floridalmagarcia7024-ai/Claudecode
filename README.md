# Polymarket Trading Bot — MVP (Phase 1)

Algorithmic trading bot for Polymarket prediction markets. Uses free AI (Groq + VADER) with a mean reversion strategy in paper trading mode.

## Quick Start (5 Steps)

### 1. Clone and install

```bash
git clone <repo-url> && cd polymarket-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys:
#   GROQ_API_KEY        — Free at https://console.groq.com
#   POLYMARKET_PRIVATE_KEY — Your Polygon wallet private key
#   WALLET_ADDRESS      — Your wallet address
```

### 3. Run in paper mode (default)

```bash
python main.py
```

The bot starts in **paper mode** by default. It will:
- Connect to Polymarket and fetch active markets
- Begin collecting market data every 5 minutes (building history for z-score)
- Generate signals once 7 days of data accumulates
- Simulate trades with realistic slippage and fees

### 4. Monitor

```bash
# Health check
curl http://localhost:8000/health

# View positions
curl http://localhost:8000/positions

# View trades
curl http://localhost:8000/trades
```

### 5. Deploy to Railway (~$5/month)

```bash
# Push to GitHub, then connect to Railway
# Set environment variables in Railway dashboard
# The bot auto-restarts on failure via health checks
```

## Architecture

```
core/engine.py        — Main trading loop (3 concurrent async tasks)
core/state.py         — SQLite persistence (positions, trades, P&L)
core/risk.py          — Trailing stop (4-state machine), circuit breaker, anti-martingale
core/paper.py         — Paper trading with simulated slippage/fees
api/polymarket.py     — Async CLOB client with rate limiting + retry
api/data_collector.py — Market snapshots every 5 min → SQLite
intelligence/         — VADER (offline) + Groq (free tier, 300 calls/day)
strategies/           — Mean reversion (z-score on 7-day probability)
```

## Strategy: Mean Reversion

- **Signal**: z-score = (current_prob - mean_7d) / std_7d
- **BUY** when z < -1.8, **SELL** when z > 1.8
- **Filters**: volume > $1,000, spread < 6%, depth > $500, slippage < 3%
- **Exit**: z-score returns to [-0.5, 0.5] or trailing stop triggers
- **Target**: 2-4 trades/day across ~50 active markets

## Risk Management

| Control | Value |
|---------|-------|
| Max daily loss | 8% of capital |
| Max position size | 10% of capital |
| Default position | 5% of capital |
| Trailing stop | 5% (after 5% breakeven trigger) |
| Anti-martingale | Reduce 15%/loss, pause at 5 consecutive |
| Category limits | Politics 40%, Crypto 30%, Sports 20%, Other 30% |

## Important Notes

- **First 7 days**: The bot collects data. No signals until history accumulates.
- **Paper mode**: All trades are simulated with 1.25% fees + variable slippage.
- **VADER**: Only works with English text. All analysis is in English.
- **Groq**: Free tier (14,400 req/day). Bot uses conservative 300/day limit.
- **No historical data API**: The bot builds its own dataset from deployment.

## Testing

```bash
pytest tests/ -v
```

## Next Phases

- **Phase 2**: RSS feeds, market regime detection, momentum strategy
- **Phase 3**: Dashboard, backtesting, Telegram alerts
- **Phase 4**: Parameter optimizer, shadow mode, cross-market intelligence
