# StonksAI 🤖📈

An AI-powered stock trading bot built with a PyTorch LSTM model, FastAPI backend, and a React dashboard. Supports Zerodha Kite Connect integration (portfolio sync) with Yahoo Finance for live price data.

---

## Project Structure

```
stonks/
│
├── .env                         ← API credentials (git-ignored)
├── .gitignore
├── global_lstm.pth              ← Trained LSTM model weights
├── watchlist.json               ← Persisted custom watchlist (auto-created)
│
├── normalize_batch.py           ← Step 1: Rolling Z-Score normalization of raw CSVs
├── train_global_model.py        ← Step 2: Trains the 3-layer LSTM on normalized data
├── live_pipeline.py             ← Standalone CLI paper-trading script (no UI needed)
│
├── normalized_data/             ← Z-Score normalized Parquet files (output of Step 1)
│
├── backend/
│   └── main.py                  ← FastAPI server
│                                   • Loads LSTM model on startup
│                                   • Integrates with Zerodha Kite Connect (optional)
│                                   • GET  /api/predict/{ticker}  → LSTM signal + candles
│                                   • GET  /api/tickers           → portfolio/default tickers
│                                   • GET  /api/watchlist         → custom watchlist
│                                   • POST /api/watchlist/{sym}   → add to watchlist
│                                   • DELETE /api/watchlist/{sym} → remove from watchlist
│                                   • GET  /api/status            → server + Kite status
│                                   • POST /api/config            → save data source config
│                                   • GET  /api/login             → Zerodha OAuth redirect
│                                   • GET  /api/callback          → Zerodha OAuth callback
│
└── frontend/
    ├── index.html               ← HTML shell with SEO meta tags
    └── src/
        ├── main.jsx             ← React entry point
        ├── index.css            ← Full dark-mode design system
        └── App.jsx              ← Main React dashboard
                                    • Setup page (choose YFinance or Zerodha)
                                    • Dynamic ticker bar (Kite holdings + custom watchlist)
                                    • 60-candle AreaChart with price history
                                    • BUY / SELL / HOLD signal badge + confidence bar
                                    • Auto Trading mode (30s scan cycle, paper P&L tracking)
                                    • Simulated trade log with entry/exit P&L
```

---

## Architecture

```
                     ┌─────────────────────────────────────────────────────┐
                     │              React Frontend (localhost:5173)         │
                     │  ┌──────────┐  ┌────────────┐  ┌────────────────┐  │
                     │  │ Setup    │  │ Ticker Bar  │  │ Trade Log /    │  │
                     │  │ Page     │  │ (dynamic)   │  │ Auto Mode      │  │
                     │  └──────────┘  └────────────┘  └────────────────┘  │
                     └───────────────────────┬─────────────────────────────┘
                                             │ HTTP (axios)
                     ┌───────────────────────▼─────────────────────────────┐
                     │            FastAPI Backend (localhost:8000)          │
                     │  ┌───────────────────────────────────────────────┐  │
                     │  │  Rolling Z-Score  →  LSTM (global_lstm.pth)  │  │
                     │  └───────────────────────────────────────────────┘  │
                     │         ▲ yfinance          ▲ Kite Connect          │
                     └─────────┼───────────────────┼───────────────────────┘
                               │                   │
                         Live price data      Portfolio holdings
                         (candle fetching)    (ticker list only)
```

---

## Data Flow

```
1. yfinance fetches 5 days of 1-min candles for the selected ticker
2. Backend applies rolling Z-Score normalization (window=1000)
3. LSTM model runs on the last 60 normalized candles
4. Returns: { signal: BUY|SELL|HOLD, confidence, current_price, candles }
5. React renders: AreaChart + Signal Badge + Confidence Bar
6. Auto Mode: repeats every 30s, tracks paper positions & P&L
```

---

## Setup

### 1 — Install dependencies

```bash
# Backend
pip install fastapi uvicorn yfinance torch numpy pandas python-dotenv kiteconnect

# Frontend
cd frontend
npm install
```

### 2 — (Already done) Normalize data & train model

```bash
python normalize_batch.py      # generates normalized_data/
python train_global_model.py   # generates global_lstm.pth
```

### 3 — Configure credentials (optional — Zerodha)

Edit `.env` in the project root:

```env
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=           # auto-filled after login
DATA_SOURCE=kite             # or "yfinance"
```

> **Note:** Zerodha Kite Personal plan only allows portfolio/holdings access.
> Historical candle data (`historical_data` API) requires the Connect plan (₹2000/month).
> The app runs in **hybrid mode** — Kite for your portfolio ticker list, YFinance for price data.

### 4 — Start the backend

```bash
cd backend
& "C:\...\python.exe" -m uvicorn main:app --reload --port 8000
# or simply:
uvicorn main:app --reload --port 8000
```

Test: http://localhost:8000/api/status

### 5 — Start the frontend

```bash
cd frontend
npm run dev
# Opens http://localhost:5173
```

---

## Features

| Feature | Description |
|---|---|
| 🧠 **LSTM Model** | 3-layer, 128 hidden units, trained on 537 NSE stocks |
| 📈 **Live Chart** | 60-candle AreaChart with price range highlighting |
| 🎯 **Signal Badge** | BUY / SELL / HOLD with confidence % |
| 🤖 **Auto Trading** | 30s scan cycle, paper position tracking, session P&L |
| 📋 **Trade Log** | Persistent entry/exit log with per-trade P&L |
| 🏦 **Zerodha Sync** | Auto-populates ticker bar from your Kite holdings |
| ⭐ **Custom Watchlist** | Add/remove any NSE symbol, persisted to `watchlist.json` |
| ⚙️ **Setup Page** | Choose data source, enter Zerodha credentials from the UI |

---

## Supported Tickers (defaults)

`RELIANCE`, `TCS`, `INFY`, `HDFC`, `BAJFINANCE`, `PNB`, `DWARKESH`

When Zerodha is connected, the ticker bar is automatically replaced with your actual portfolio holdings. You can also add any NSE symbol via the custom watchlist input.
