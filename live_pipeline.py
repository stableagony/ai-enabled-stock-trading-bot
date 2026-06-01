import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Model Definition (Must match the trained model) ---
class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# --- Configurations ---
TICKER = "RELIANCE.NS" # Yahoo Finance ticker for Reliance
MODEL_PATH = "global_lstm.pth"
SEQUENCE_LENGTH = 60
WINDOW = 1000 # Same rolling window used in training
POLLING_INTERVAL = 10 # seconds (Shortened for demonstration)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading Model on {device}...")

# Load the model
model = TradingLSTM(input_size=5, hidden_size=128, num_layers=3, output_size=1).to(device)

def fetch_and_predict():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching latest live data for {TICKER} from yfinance...")
    
    # Fetch last 5 days of minute data to have enough history for the Z-score rolling window
    data = yf.download(TICKER, period="5d", interval="1m", progress=False)
    
    if len(data) < SEQUENCE_LENGTH:
        print("Not enough data fetched.")
        return
        
    # Formatting the data
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single tickers in newer versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Store the actual current price for our log
    current_price = df['close'].iloc[-1]
    
    # Apply Rolling Z-Score Normalization
    current_window = min(WINDOW, len(df))
    for col in df.columns:
        rolling_mean = df[col].rolling(window=current_window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=current_window, min_periods=1).std()
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
    # Get the last 60 minutes for our sequence
    recent_data = df.iloc[-SEQUENCE_LENGTH:].values
    
    # Convert to tensor: shape [1, 60, 5]
    input_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        # Because we used BCEWithLogitsLoss during training, the output is raw logits. We need sigmoid for probability.
        probability = torch.sigmoid(output).item()
        
    print(f"Current Market Price: ₹{current_price:.2f}")
    if probability > 0.55:
        print(f"🤖 Prediction: UP (Confidence: {probability*100:.1f}%) -> [ACTION] SIMULATED BUY TRIGGERED")
    elif probability < 0.45:
        print(f"🤖 Prediction: DOWN (Confidence: {(1-probability)*100:.1f}%) -> [ACTION] SIMULATED SELL/SHORT TRIGGERED")
    else:
        print(f"🤖 Prediction: NEUTRAL (Confidence: {probability*100:.1f}%) -> [ACTION] HOLDING (No trade)")

if __name__ == "__main__":
    print("Waiting for global_lstm.pth to be available...")
    while not os.path.exists(MODEL_PATH):
        time.sleep(2)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    
    print("Starting Live Paper Trading Pipeline Simulation...")
    try:
        # Run 3 times to simulate for the user, then stop automatically
        for _ in range(3):
            fetch_and_predict()
            print(f"Waiting {POLLING_INTERVAL} seconds for next candle...")
            time.sleep(POLLING_INTERVAL)
        print("\nSimulation complete. In a real environment, this loop runs continuously.")
    except KeyboardInterrupt:
        print("\nStopping Live Pipeline.")
