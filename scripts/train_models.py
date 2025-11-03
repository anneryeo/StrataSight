"""
Convenience script to train both models for a ticker and save them to models/saved/
"""

from pathlib import Path
from src.data_fetch import fetch_tickers
from src.models.lstm_model import train_and_save as train_lstm
from src.models.prophet_model import train_and_save as train_prophet

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    tickers = ['TSLA', 'GME', 'AMD', 'AAPL']
    data = fetch_tickers(tickers, period='1y')
    for t in tickers:
        print(f'Training for {t}...')
        df = data[t]
        train_lstm(df, str(MODEL_DIR / f'{t}_lstm'))
        train_prophet(df, str(MODEL_DIR / f'{t}_prophet.joblib'))
        print(f'Done: {t}')
