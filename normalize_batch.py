import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# Suppress performance warnings from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

INPUT_DIR = 'data'
OUTPUT_DIR = 'normalized_data'
WINDOW = 1000  # Approx 2.5 trading days of minute data (assuming 375 minutes per trading day in India)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(file_path):
    try:
        filename = os.path.basename(file_path)
        out_name = filename.replace('.csv', '.parquet')
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Skip if already exists to allow resuming if interrupted
        if os.path.exists(out_path):
            return True
            
        # Read the csv
        df = pd.read_csv(file_path)
        
        # Check if date column exists
        if 'date' not in df.columns:
            print(f"Skipping {filename}: 'date' column not found.")
            return False
            
        # Convert date and set index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
        
        # Rolling Z-score to prevent data leakage from future data
        for col in cols_to_normalize:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=WINDOW).mean()
                rolling_std = df[col].rolling(window=WINDOW).std()
                # add a tiny epsilon to avoid division by zero
                df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                
        # Drop rows with NaN (the first WINDOW-1 rows will be NaN because of the rolling window)
        df.dropna(inplace=True)
        
        # Save to parquet. pyarrow is very fast.
        df.to_parquet(out_path, engine='pyarrow')
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

if __name__ == '__main__':
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    
    print(f"Found {len(csv_files)} files. Starting Z-Score Normalization (Rolling Window={WINDOW})...")
    
    # Process in parallel using limited CPU cores to prevent OOM
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_file, csv_files), total=len(csv_files)))
        
    successes = sum(1 for r in results if r)
    print(f"Successfully normalized and saved {successes}/{len(csv_files)} files to {OUTPUT_DIR}/")
