# BTC Price Forecasting with RNNs

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to forecast
the closing price of Bitcoin (BTC/USD) one hour into the future, using the
past 24 hours of price data.

## Dataset
- Source: Coinbase BTC/USD historical trade data
- File: `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- Each row represents a 60-second trading window
- Total rows: 2,099,760 (Dec 2014 → Jan 2019)

## Files
| File | Description |
|------|-------------|
| `preprocess_data.py` | Cleans, normalizes and saves the dataset |
| `forecast_btc.py` | Builds, trains and evaluates the LSTM model |
| `README.md` | Project documentation |

## Preprocessing (`preprocess_data.py`)
### Steps
1. Load raw CSV and sort by timestamp
2. Add cyclic time features (hour_sin, hour_cos, day_sin, day_cos)
3. Select features: Open, High, Low, Close + 4 cyclic features
4. Forward-fill missing rows (inactive trading windows)
5. Chronological split: 80% train / 10% val / 10% test
6. MinMax normalization fitted on train only, clipped to [0, 1]
7. Save scaled arrays as `.npy` files

### Feature Decisions
| Feature | Decision | Reason |
|---------|----------|--------|
| Open, High, Low, Close | Keep | Core price signals |
| Volume_(BTC) | Drop | Low correlation (0.15), very noisy |
| Volume_(Currency) | Drop | Collinear with Volume_(BTC) |
| Weighted_Price | Drop | Near-duplicate of Close (corr=1.0) |
| Timestamp | Encode | Converted to sin/cos cyclic features |

### Missing Data
- 109,069 rows (5.19%) have all values as NaN
- These represent inactive trading windows, not corrupted data
- Strategy: forward-fill (carry last known price forward)

## Model (`forecast_btc.py`)

### Architecture
Input: (1440, 8) → 24 hours × 8 features
LSTM(64, return_sequences=True)
LSTM(32)
Dense(1) → predicted Close price

### Training
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Early stopping: patience=3
- Data pipeline: tf.data.Dataset with batch=512 and prefetch

### Results
| Set | MSE |
|-----|-----|
| Validation | 4.1e-07 |
| Test | 3.7e-06 |

## How to Run

### 1. Preprocess
```bash
python preprocess_data.py
```

### 2. Train and evaluate
```bash
python forecast_btc.py
```

## Requirements
tensorflow
numpy
pandas

## Author
Arber Zylyftari - Holberton School