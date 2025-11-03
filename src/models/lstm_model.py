"""Minimal LSTM training helper using TensorFlow/Keras."""
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


def create_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_series(df: pd.DataFrame, target_col: str = 'Close', window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    # Handle DataFrames that may have MultiIndex columns (yfinance returns
    # MultiIndex when multiple tickers are requested). If so, try to find a
    # column whose second level matches the target_col (e.g. ('TSLA','Close')).
    if isinstance(df.columns, pd.MultiIndex):
        # prefer an exact match like (ticker, 'Close') if present
        matches = [c for c in df.columns if c[1] == target_col]
        if not matches:
            raise KeyError(f"Target column '{target_col}' not found in MultiIndex columns")
        series = df[matches[0]].values.astype('float32')
    else:
        if target_col in df.columns:
            series = df[target_col].values.astype('float32')
        else:
            # fallback: use first numeric column
            series = df.select_dtypes(include=["number"]).iloc[:, 0].values.astype('float32')
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window]) 
        y.append(series[i+window])
    X = np.array(X)[:, :, np.newaxis]
    y = np.array(y)[:, np.newaxis]
    return X, y


def train_and_save(df: pd.DataFrame, out_path: str, epochs: int = 5, window: int = 20):
    X, y = prepare_series(df, window=window)
    model = create_lstm_model(input_shape=X.shape[1:])
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    # Ensure a proper file extension for Keras model saving. Keras requires
    # either a `.keras` (native format) or `.h5` extension when saving to a
    # single file. If the caller provided a directory-like path without
    # extension, append `.keras`.
    p = Path(out_path)
    if not p.suffix:
        out_path = str(p.with_suffix('.keras'))
        p = Path(out_path)

    os.makedirs(p.parent, exist_ok=True)
    model.save(out_path)
    return out_path
