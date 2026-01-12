import pandas as pd
import numpy as np
import torch
import os

def normalize_index(df):
    """Aligns and normalizes the datetime index."""
    idx = pd.to_datetime(df.index, errors='coerce')
    if idx.isna().all():
        return df
    mask = ~idx.isna()
    df = df.loc[mask].copy()
    df.index = idx[mask].normalize()
    return df[~df.index.duplicated(keep='last')].sort_index()

def compute_features(df):
    """
    Computes financial features for GA embedding:
    - Log Returns
    - Volatility (20d)
    - Momentum (10d)
    - Volume Ratio
    """
    df = df.copy()
    if 'close' not in df.columns:
        return df
        
    # Returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility_20d'] = df['return'].rolling(window=20).std() * np.sqrt(252)
    
    # Momentum
    df['momentum_10d'] = df['close'].pct_change(10)
    
    # Volume Ratio
    if 'volume' in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 0.0
        
    return df.dropna()

def load_and_process(symbol="QQQ"):
    """
    Loads raw data, processes it, and returns a PyTorch tensor.
    """
    raw_path = os.path.join("data", "raw", f"{symbol}_historical.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data for {symbol} not found. Run fetch_data.py first.")
    
    df = pd.read_csv(raw_path, index_col=0)
    df = normalize_index(df)
    df = compute_features(df)
    
    # Select features for embedding
    feature_cols = ['log_return', 'volatility_20d', 'momentum_10d', 'volume_ratio']
    tensor_data = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    return df.index, tensor_data

if __name__ == "__main__":
    try:
        idx, data = load_and_process("QQQ")
        print(f"Processed QQQ data: {data.shape} tensor created.")
        output_path = os.path.join("data", "processed", "QQQ_features.pt")
        torch.save(data, output_path)
        print(f"Saved tensor to {output_path}")
    except Exception as e:
        print(e)
