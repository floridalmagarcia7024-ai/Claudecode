# CLAUDE.md — Polymarket Bot Session Context

Lee este archivo PRIMERO antes de hacer cualquier cambio en una nueva sesión.

## Resumen del Proyecto
Bot de trading para Polymarket (mercados de predicción). 4 fases completas.
- **Runtime**: Python 3.11+, FastAPI, uvicorn
- **Dashboard**: http://localhost:8000 (login: admin / polymarket)
- **Modos**: Paper (simulado) y Real (Polymarket CLOB API)

## Arquitectura
main.py # Entry point + lifespan startup
config.py # Config vía pydantic-settings (lee .env)
├── api/ # Cliente CLOB + data collector
├── core/ # Engine, paper, risk, state, regime, stress test
├── strategies/ # Estrategias (todas extienden BaseStrategy)
├── intelligence/ # AI (Groq+VADER), RSS news, cross-market, order flow
├── execution/ # SmartExit (trailing stop, slippage)
├── backtesting/ # Backtester + walk-forward optimizer
├── optimizer/ # Auto-optimizer, shadow bots
├── monitoring/ # Health monitor, audit logger
├── journal/ # Trade recorder + AI journal
├── telegram_bot/ # Bot Telegram (comandos + alertas)
├── ab_test/ # A/B test manager
└── dashboard/
├── main.py # Sub-app FastAPI con todos los endpoints REST
└── templates/ # index.html, onboarding.html, calibration.html

## Historial de Fases
| Fase | Branch | Qué se construyó |
|------|--------|------------------|
| 1 MVP | claude/implement-mvp-phase-one-gCETI | Engine, paper/real, 2 estrategias, state |
| 2 Intelligence | claude/polymarket-bot-phase-2-9FUCv | AI, news RSS, regime, telegram, health |
| 3 UX | claude/polymarket-bot-phase-3-bbYxx | Dashboard, backtesting, smart exit, journal |
| 4 Advanced | claude/polymarket-phase-4-89k7W | Optimizer, shadow, stress, cross-market, A/B |
| 5 Fix/UX | claude/fix-polymarket-bot-rW0v5 | UX fixes, log stream, credentials API, +3 estrategias |

## Estrategias (strategies/)
Todas extienden `BaseStrategy` de `strategies/base.py`.

| Archivo | Clase | Señal |
|---------|-------|-------|
| mean_reversion.py | MeanReversionStrategy | z-score > 1.8 en historial 7d |
| momentum.py | MomentumStrategy | Cambio 1h ≥ 2.5% + régimen trending |
| news_surge.py | NewsSurgeStrategy | Surge de sentimiento VADER + volumen alto |
| value_bet.py | ValueBetStrategy | Edge AI vs precio mercado ≥ 10% |
| liquidity_squeeze.py | LiquiditySqueezeStrategy | Spread se ensancha 3x → revertir |

**Para agregar una estrategia nueva:**
1. Crear `strategies/tu_estrategia.py` implementando `BaseStrategy`
2. Importar e instanciar en el lifespan de `main.py`
3. Agregar a la lista `strategies` que pasa a `TradingEngine`
4. Agregar checkbox en `onboarding.html`

## Variables .env Clave
PAPER_MODE=true
POLYMARKET_API_KEY=
POLYMARKET_SECRET=
WALLET_ADDRESS=0x...
POLYMARKET_PRIVATE_KEY=
GROQ_API_KEY=gsk_... # Free: https://console.groq.com
TELEGRAM_BOT_TOKEN= # De @BotFather
TELEGRAM_CHAT_ID=
DASHBOARD_USER=admin
DASHBOARD_PASS=polymarket
JWT_SECRET= # Auto-generado si está vacío

## Endpoints API (todos requieren JWT excepto /health y /api/auth/login)
GET /health
POST /api/auth/login { username, password } → { access_token }
GET /api/status
POST /api/start
POST /api/stop
POST /api/mode/toggle Alterna paper ↔ real
GET /api/positions
POST /api/positions/{id}/close
GET /api/trades
GET /api/metrics
POST /api/config { key, value } — un parámetro a la vez
GET /api/settings Config actual (credenciales enmascaradas)
POST /api/settings/credentials Guarda API keys al .env
GET /api/logs Entradas recientes (ring buffer memoria)
GET /api/logs/stream SSE stream en vivo
GET /api/export/trades
GET /api/backtest/{strategy}
GET 
## Ejecutar Localmente
```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env — mínimo: PAPER_MODE=true
python main.py
# Abrir http://localhost:8000 → admin / polymarket
/api/regime/{market_id}

Síntoma	Causa	Solución
Engine "503 not available"	Sin API keys	Poner PAPER_MODE=true o agregar keys
Dashboard en blanco tras login	JWT expirado	Logout + re-login
"No data" en métricas	Sin trades aún	Correr en paper mode primero
Estrategia no dispara	Filtros muy estrictos	Bajar zscore_threshold con /api/config
Logs no aparecen	Ring buffer vacío al iniciar	Esperar que el bot procese mercados

