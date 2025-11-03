"""Minimal LSTM training helper using TensorFlow/Keras."""
import os
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
    series = df[target_col].values.astype('float32')
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    return out_path
