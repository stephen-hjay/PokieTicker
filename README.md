# PokieTicker — Stock News Analysis Platform

Real-time stock candlestick charts with AI-powered news analysis. Hover over any trading day to instantly see relevant news, sentiment, and price impact factors.

## Architecture

```
Frontend (React + Vite + D3.js)          Backend (FastAPI + SQLite)
┌─────────────────────────────┐          ┌──────────────────────────┐
│  CandlestickChart (D3.js)   │──hover──▶│  /api/stocks/{sym}/ohlc  │
│  ├─ crosshair follows mouse │          │  /api/news/{sym}?date=   │
│  └─ news dots on dates      │          │  /api/news/{sym}/timeline│
│                              │          │                          │
│  NewsPanel (right sidebar)   │◀─JSON───│  SQLite: pokieticker.db  │
│  ├─ sentiment sorted         │          │  ├─ ohlc                 │
│  ├─ up/down reasons          │          │  ├─ news_raw             │
│  └─ T+1/T+5 returns         │          │  └─ layer1_results       │
└─────────────────────────────┘          └──────────────────────────┘
```

## Data Pipeline

```
Polygon API ──▶ Layer 0 (rule filter, free) ──▶ Layer 1 (Haiku Batch API) ──▶ Layer 2 (Sonnet, on-demand)
  OHLC + News     reject empty/spam/listicles    50 articles per API call       deep analysis on click
                   ~17% rejected                  sentiment + up/down reasons    cached in DB
```

## Quick Start

### 1. Install dependencies

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

- **Polygon.io** — free tier at [polygon.io](https://polygon.io/)
- **Anthropic** — sign up at [console.anthropic.com](https://console.anthropic.com/)

### 3. Initialize database

```bash
python -m backend.database
```

### 4. Run

```bash
# Terminal 1: Backend
uvicorn backend.api.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**

## Project Structure

```
.env                          # API keys (gitignored)
pokieticker.db                # SQLite database (gitignored)
requirements.txt              # Python dependencies

backend/
  config.py                   # pydantic-settings, loads .env
  database.py                 # 9-table SQLite schema
  migration.py                # One-time CSV/JSONL → SQLite import
  bulk_fetch.py               # Bulk download OHLC + news for many tickers
  batch_submit.py             # Submit Layer 1 to Anthropic Batch API
  batch_collect.py            # Collect Batch API results
  weekly_update.py            # Incremental weekly data update
  polygon/
    client.py                 # Polygon API with retry/backoff
  pipeline/
    layer0.py                 # Rule-based filter (free)
    layer1.py                 # Claude Haiku batch analysis
    layer2.py                 # Sonnet on-demand deep analysis
    alignment.py              # News → trading day + forward returns
  ml/
    features.py               # Feature engineering
    model.py                  # ML model training
    inference.py              # Model inference
    backtest.py               # Backtesting framework
  api/
    main.py                   # FastAPI app + CORS
    routers/
      stocks.py               # GET /api/stocks, /search, /{sym}/ohlc
      news.py                 # GET /api/news/{sym}, /{sym}/timeline
      analysis.py             # POST /api/analysis/deep, /story
      pipeline.py             # POST /api/pipeline/fetch, /process
      predict.py              # GET /api/predict

frontend/
  src/
    App.tsx                   # Main layout (chart + news sidebar)
    App.css                   # Dark theme styling
    components/
      StockSelector.tsx       # Ticker tabs + search
      CandlestickChart.tsx    # D3.js chart with crosshair
      NewsPanel.tsx           # Sentiment-sorted news cards
      StoryPanel.tsx          # AI story generation
```

## Data Operations

### Add new tickers

```bash
# Bulk fetch (handles rate limiting automatically)
python -m backend.bulk_fetch
```

### Run Layer 1 analysis (Anthropic Batch API)

```bash
# Submit batch (~$0.35 per 1000 articles)
python -m backend.batch_submit --top 50

# Check status & collect results
python -m backend.batch_collect <batch_id>
```

### Weekly incremental update

```bash
# Manual
python -m backend.weekly_update

# Cron (every Sunday 2am)
# crontab -e, then add:
# 0 2 * * 0 cd /path/to/PokieTicker && python -m backend.weekly_update >> logs/weekly.log 2>&1
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List all tickers |
| GET | `/api/stocks/search?q=` | Search tickers |
| GET | `/api/stocks/{sym}/ohlc` | OHLC data for chart |
| POST | `/api/stocks` | Add new ticker |
| GET | `/api/news/{sym}?date=` | News for a trading day |
| GET | `/api/news/{sym}/timeline` | Dates with news (chart markers) |
| POST | `/api/analysis/deep` | Layer 2 deep analysis |
| POST | `/api/analysis/story` | AI trend story generation |
| POST | `/api/pipeline/fetch` | Trigger Polygon data fetch |
| POST | `/api/pipeline/process` | Run Layer 0 + Layer 1 |

## Cost Summary

| Item | Cost |
|------|------|
| Polygon data (free tier) | $0 |
| Layer 1 Batch API (per 1000 articles) | ~$0.35 |
| Layer 2 on-demand (per article) | ~$0.003 |
| Weekly incremental update | ~$1-2 |

## Tech Stack

- **Frontend**: React, TypeScript, Vite, D3.js, Axios
- **Backend**: FastAPI, SQLite (WAL mode), Pydantic
- **AI**: Claude Haiku 4.5 (batch), Claude Sonnet (on-demand)
- **Data**: Polygon.io REST API

## License

MIT — see [LICENSE](LICENSE) for details.
