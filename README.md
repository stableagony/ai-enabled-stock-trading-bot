# StonksAI — How All Files Connect

```
stonks/
│
├── data/                        ← Raw minute-level CSV files (537 stocks, ~21 GB)
│
├── normalized_data/             ← Z-Score normalized Parquet files (output of normalize_batch.py)
│
├── normalize_batch.py           ← Step 1: Reads data/, applies rolling Z-Score, saves to normalized_data/
│
├── train_global_model.py        ← Step 2: Reads normalized_data/, trains the PyTorch LSTM model
│                                          Saves weights → global_lstm.pth
│
├── global_lstm.pth              ← The trained model weights (created after Step 2)
│
├── live_pipeline.py             ← Standalone paper-trading script (no web UI needed)
│                                  Loads global_lstm.pth, fetches live yfinance data, prints BUY/SELL/HOLD
│
├── backend/
│   └── main.py                  ← Step 3: FastAPI server
│                                  • Loads global_lstm.pth on startup
│                                  • GET /api/predict/{ticker}
│                                    → Fetches live data via yfinance
│                                    → Normalizes it on the fly (same rolling Z-Score)
│                                    → Runs the LSTM model
│                                    → Returns JSON: { signal, confidence, candles, current_price }
│                                  • GET /api/tickers  → list of supported tickers
│                                  • GET /health       → server health check
│
├── frontend/
│   ├── index.html               ← HTML shell (title, meta SEO tags)
│   └── src/
│       ├── main.jsx             ← React entry point — mounts <App /> into #root
│       ├── index.css            ← Full dark-mode design system (tokens, animations, layout)
│       └── App.jsx              ← Step 4: The entire React dashboard UI
│                                  • Calls backend GET /api/predict/{ticker}
│                                  • Displays an AreaChart (Recharts) of 60-min price history
│                                  • Shows BUY/SELL/HOLD signal badge + confidence bar
│                                  • Logs every prediction to a simulated trade log
│
├── model_training.ipynb         ← Jupyter notebook for single-stock LSTM experiments
└── initializing.ipynb           ← Data loading + basic feature engineering notebook
```

## How to Run Everything

### 1 — Normalize the data (already done)
```bash
python normalize_batch.py
```

### 2 — Train the model (already done / in progress)
```bash
python train_global_model.py
# produces global_lstm.pth
```

### 3 — Start the FastAPI backend (Terminal 1)
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Test it: http://localhost:8000/api/predict/RELIANCE

### 4 — Start the React frontend (Terminal 2)
```bash
cd frontend
npm run dev
# Opens http://localhost:5173
```

The frontend calls `http://localhost:8000` → the backend loads `global_lstm.pth`
and returns predictions → the UI renders the chart + signal.

## Data Flow Diagram

```
yfinance (live)
     ↓
backend/main.py
  rolling Z-Score normalization
     ↓
global_lstm.pth  (PyTorch LSTM)
     ↓
JSON { signal, confidence, candles }
     ↓
React App.jsx
  AreaChart + Signal Badge + Trade Log
```
