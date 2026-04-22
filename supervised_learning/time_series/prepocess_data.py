#!/usr/bin/env python3
"""Preprocesses raw Coinbase BTC/USD 1-min data for RNN forecasting."""

import numpy as np
import pandas as pd


def load_and_clean(path):
    """Loading CSV, sort by time, forward-fill missing rows."""
    df = pd.read_csv(path)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df[['Open', 'High', 'Low', 'Close']]
    df = df.ffill().dropna()
    return df


def normalize(train, val, test):
    """MinMax scale fitted on train only. Returns scaled arrays + scaler params."""
    data_min = train.min(axis=0)
    data_max = train.max(axis=0)

    def scale(arr):
        return (arr - data_min) / (data_max - data_min)

    return scale(train), scale(val), scale(test), data_min, data_max


def make_windows(data, window=1440):
    """Sliding windows: 1440 timesteps in : next Close as target."""
    X, y = [], []
    data = data.values
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window, 3])  # index 3 = Close
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    path = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

    print("Loading and cleaning data...")
    df = load_and_clean(path)
    print(f"Clean rows: {len(df):,}")

    # Chronological split: 80 / 10 / 10
    n = len(df)
    t1 = int(n * 0.80)
    t2 = int(n * 0.90)

    train = df.iloc[:t1].values.astype(np.float32)
    val   = df.iloc[t1:t2].values.astype(np.float32)
    test  = df.iloc[t2:].values.astype(np.float32)

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("Normalizing (fit on train only)...")
    train_s, val_s, test_s, d_min, d_max = normalize(train, val, test)

    print("Building sliding windows (this takes a moment)...")
    X_train, y_train = make_windows(train_s)
    X_val,   y_val   = make_windows(val_s)
    X_test,  y_test  = make_windows(test_s)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val   shape: {X_val.shape}")
    print(f"X_test  shape: {X_test.shape}")

    print("Saving .npz files...")
    np.savez('train.npz', X=X_train, y=y_train)
    np.savez('val.npz',   X=X_val,   y=y_val)
    np.savez('test.npz',  X=X_test,  y=y_test)
    np.savez('scaler.npz', data_min=d_min, data_max=d_max)

    print("Done. Files saved: train.npz, val.npz, test.npz, scaler.npz")


if __name__ == '__main__':
    main()