import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# ── Load environment variables ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
DATA_SOURCE = os.getenv("DATA_SOURCE", "yfinance")  # "kite" or "yfinance"

# ── Kite Connect (optional) ─────────────────────────────────────────────────
kite = None

def init_kite():
    global kite
    if not KITE_API_KEY or KITE_API_KEY == "your_api_key_here":
        print("[INFO] Kite API key not configured -- using yfinance as data source.")
        return False
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            kite.set_access_token(KITE_ACCESS_TOKEN)
            print("[OK] Kite Connect initialized with existing access token.")
            return True
        else:
            print("[INFO] Kite client created but no access token. Visit /api/login to authenticate.")
            return True
    except ImportError:
        print("[WARN] kiteconnect package not installed -- using yfinance.")
        return False
    except Exception as e:
        print(f"[WARN] Kite init failed: {e} -- using yfinance.")
        return False

# ── NSE Instrument Token Cache ───────────────────────────────────────────────
instrument_cache = {}

def load_instruments():
    global instrument_cache
    if kite and kite.access_token:
        try:
            instruments = kite.instruments("NSE")
            for inst in instruments:
                instrument_cache[inst["tradingsymbol"]] = inst["instrument_token"]
            print(f"[OK] Loaded {len(instrument_cache)} NSE instrument tokens.")
        except Exception as e:
            print(f"[WARN] Failed to load instruments: {e}")

# ── LSTM model (must match train_global_model.py) ───────────────────────────
class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).squeeze()

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "global_lstm.pth")
SEQUENCE_LENGTH = 60
WINDOW = 1000

model: Optional[TradingLSTM] = None

def load_model():
    global model
    m = TradingLSTM(input_size=5, hidden_size=128, num_layers=3, output_size=1)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        m.eval()
        print("[OK] Model loaded from", MODEL_PATH)
    else:
        print("[WARN] global_lstm.pth not found -- predictions will use an untrained model.")
    model = m

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="StonksAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stableagony.github.io",
        "https://stableagony.github.io/ai-enabled-stock-trading-bot",
        "https://stableagony.github.io/ai-enabled-stock-trading-bot/",
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    try:
        load_model()
    except Exception as e:
        print(f"[WARN] Model load failed: {e}")
    try:
        if init_kite():
            load_instruments()
    except Exception as e:
        print(f"[WARN] Kite init failed: {e}")

# ── Pydantic schemas ─────────────────────────────────────────────────────────
class CandlePoint(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    confidence: float
    signal: str
    candles: List[CandlePoint]
    currency: str
    data_source: str  # "kite" or "yfinance"

class ConfigPayload(BaseModel):
    data_source: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

DEFAULT_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFC", "BAJFINANCE", "PNB", "DWARKESH"]

YF_SYMBOLS = {
    "HDFC": "HDFCBANK.NS",
}

def rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    w = min(WINDOW, len(df))
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            rm = df[col].rolling(window=w, min_periods=1).mean()
            rs = df[col].rolling(window=w, min_periods=1).std()
            df[col] = (df[col] - rm) / (rs + 1e-8)
    return df

# ── Data Fetchers ─────────────────────────────────────────────────────────────
def fetch_via_yfinance(ticker_key: str) -> pd.DataFrame:
    yf_sym = YF_SYMBOLS.get(ticker_key, ticker_key + ".NS")
    raw = yf.download(yf_sym, period="5d", interval="1m", progress=False, auto_adjust=True)
    if raw is None or len(raw) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=503, detail=f"Not enough data from yfinance for {yf_sym}.")
    # Flatten MultiIndex columns if present (yfinance >= 0.2.x returns MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    # Normalize column names to lowercase
    raw.columns = [c.lower() for c in raw.columns]
    needed = [c for c in ['open','high','low','close','volume'] if c in raw.columns]
    df = raw[needed].copy()
    df.columns = ['open','high','low','close','volume'][:len(needed)]
    return df.ffill().dropna()

def fetch_via_kite(ticker_key: str) -> pd.DataFrame:
    if not kite or not kite.access_token:
        raise HTTPException(status_code=503, detail="Kite not authenticated. Visit /api/login first.")
    
    nse_sym = "HDFCBANK" if ticker_key == "HDFC" else ticker_key
    token = instrument_cache.get(nse_sym)
    if not token:
        raise HTTPException(status_code=404, detail=f"Instrument token not found for {nse_sym}. Try refreshing instruments.")
    
    now = datetime.now()
    from_date = now - timedelta(days=5)
    
    try:
        records = kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=now.strftime("%Y-%m-%d"),
            interval="minute"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Kite API error: {str(e)}")
    
    if not records or len(records) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=503, detail="Not enough data from Kite Connect.")
    
    df = pd.DataFrame(records)
    df = df.rename(columns={"date": "datetime"})
    df.set_index("datetime", inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df[['open','high','low','close','volume']].ffill().dropna()

def fetch_data(ticker_key: str) -> tuple:
    """Returns (DataFrame, data_source_name).
    Always uses yfinance for candle data (Kite Personal plan lacks historical_data API).
    Kite is used only for portfolio/holdings."""
    source = "yfinance"
    if kite and kite.access_token:
        source = "yfinance (Kite portfolio)"
    return fetch_via_yfinance(ticker_key), source

# ── Zerodha OAuth Login Flow ─────────────────────────────────────────────────
@app.get("/api/login")
def kite_login():
    """Redirect user to Zerodha login page."""
    if not KITE_API_KEY or KITE_API_KEY == "your_api_key_here":
        raise HTTPException(status_code=400, detail="KITE_API_KEY not configured in .env file.")
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={KITE_API_KEY}"
    return RedirectResponse(url=login_url)

@app.get("/api/callback")
def kite_callback(request_token: str = Query(...)):
    """Handle Zerodha OAuth callback and generate access token."""
    global kite, DATA_SOURCE
    if not kite:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
    
    try:
        session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = session["access_token"]
        kite.set_access_token(access_token)
        
        # Update .env file with new token
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
            with open(env_path, "w") as f:
                for line in lines:
                    if line.startswith("KITE_ACCESS_TOKEN="):
                        f.write(f"KITE_ACCESS_TOKEN={access_token}\n")
                    elif line.startswith("DATA_SOURCE="):
                        f.write("DATA_SOURCE=kite\n")
                    else:
                        f.write(line)
        
        DATA_SOURCE = "kite"
        load_instruments()
        
        # Redirect back to frontend dashboard
        return RedirectResponse(url="http://localhost:5173?login=success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.get("/api/predict/{ticker}", response_model=PredictResponse)
def predict(ticker: str):
    ticker_key = ticker.upper()
    df, source = fetch_data(ticker_key)
    
    current_price = float(df['close'].iloc[-1])

    # Build candle payload BEFORE normalization (raw prices)
    candle_df = df.iloc[-60:].copy()
    candles = [
        CandlePoint(
            time=str(idx),
            open=round(float(row['open']), 2),
            high=round(float(row['high']), 2),
            low=round(float(row['low']), 2),
            close=round(float(row['close']), 2),
            volume=round(float(row['volume']), 2),
        )
        for idx, row in candle_df.iterrows()
    ]

    # Normalize and predict
    norm_df = rolling_zscore(df.copy())
    seq = norm_df.iloc[-SEQUENCE_LENGTH:].values
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    if prob > 0.55:
        signal = "BUY"
    elif prob < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    return PredictResponse(
        ticker=ticker_key,
        current_price=current_price,
        confidence=round(prob, 4),
        signal=signal,
        candles=candles,
        currency="INR",
        data_source=source,
    )

@app.get("/api/tickers")
def get_tickers():
    if kite and kite.access_token:
        try:
            holdings = kite.holdings()
            symbols = [h["tradingsymbol"] for h in holdings]
            if symbols:
                return symbols
        except Exception as e:
            print(f"[WARN] Failed to fetch kite holdings: {e}")
            
    return DEFAULT_TICKERS

# ── Watchlist ────────────────────────────────────────────────────────────────
WATCHLIST_FILE = os.path.join(BASE_DIR, "watchlist.json")

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                wl = json.load(f)
                if isinstance(wl, list) and len(wl) > 0:
                    return wl
        except Exception:
            pass
    return ["RELIANCE", "TCS", "INFY"]

def save_watchlist(wl):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(wl, f)
    except Exception as e:
        print(f"[WARN] Cannot write watchlist (read-only fs?): {e}")

@app.get("/api/watchlist")
def get_watchlist():
    return load_watchlist()

@app.post("/api/watchlist/{ticker}")
def add_watchlist(ticker: str):
    wl = load_watchlist()
    t = ticker.upper().strip()
    if t and t not in wl:
        wl.append(t)
        save_watchlist(wl)
    return wl

@app.delete("/api/watchlist/{ticker}")
def remove_watchlist(ticker: str):
    wl = load_watchlist()
    t = ticker.upper().strip()
    if t in wl:
        wl.remove(t)
        save_watchlist(wl)
    return wl

@app.get("/api/status")
def status():
    kite_linked = kite is not None and kite.access_token is not None if kite else False
    return {
        "model_loaded": model is not None,
        "data_source": "yfinance",
        "kite_linked": kite_linked,
        "kite_api_key_set": bool(KITE_API_KEY and KITE_API_KEY != "your_api_key_here"),
        "instruments_loaded": len(instrument_cache),
        "is_configured": True,
        "mode": "Kite Portfolio + YFinance Data" if kite_linked else "YFinance Only",
    }

@app.post("/api/config")
def update_config(payload: ConfigPayload):
    global DATA_SOURCE, KITE_API_KEY, KITE_API_SECRET
    
    env_path = os.path.join(BASE_DIR, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
            
    env_dict = {}
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env_dict[k.strip()] = v.strip()
            
    env_dict["DATA_SOURCE"] = payload.data_source
    if payload.api_key:
        env_dict["KITE_API_KEY"] = payload.api_key
    if payload.api_secret:
        env_dict["KITE_API_SECRET"] = payload.api_secret
        
    with open(env_path, "w") as f:
        f.write("# Zerodha Kite Connect Credentials\n")
        f.write("# Fill these in after creating your app at https://developers.kite.trade/\n\n")
        f.write(f"KITE_API_KEY={env_dict.get('KITE_API_KEY', '')}\n")
        f.write(f"KITE_API_SECRET={env_dict.get('KITE_API_SECRET', '')}\n\n")
        f.write("# This is set automatically after login via /api/login\n")
        f.write("# You can also paste it manually each morning\n")
        f.write(f"KITE_ACCESS_TOKEN={env_dict.get('KITE_ACCESS_TOKEN', '')}\n\n")
        f.write('# Data source: "kite" or "yfinance" (fallback if Kite is not configured)\n')
        f.write(f"DATA_SOURCE={payload.data_source}\n")

    DATA_SOURCE = payload.data_source
    if payload.api_key:
        KITE_API_KEY = payload.api_key
    if payload.api_secret:
        KITE_API_SECRET = payload.api_secret
        
    init_kite()
    
    response_data = {"status": "ok"}
    
    if payload.data_source == "kite":
        if not KITE_API_KEY or KITE_API_KEY == "your_api_key_here":
            raise HTTPException(status_code=400, detail="Kite API key is required.")
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={KITE_API_KEY}"
        response_data["redirect_url"] = login_url
        
    return response_data

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
