"""Minimal Prophet training helper."""
import os
import pandas as pd
from prophet import Prophet


def prepare_prophet_df(df: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
    df_prophet = df.reset_index()[['Date' if 'Date' in df.reset_index().columns else df.reset_index().name, target_col]]
    # ensure columns named ds and y
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    return df_prophet


def train_and_save(df: pd.DataFrame, out_path: str):
    # Prepare a DataFrame with columns ['ds', 'y'] for Prophet.
    # Handle cases where df has MultiIndex columns (yfinance returns MultiIndex
    # when multiple tickers are requested). Extract the Close series robustly.
    if isinstance(df.columns, pd.MultiIndex):
        matches = [c for c in df.columns if c[1] == 'Close']
        if not matches:
            raise KeyError("Could not find a 'Close' column in MultiIndex columns")
        series = df[matches[0]]
    else:
        if 'Close' in df.columns:
            series = df['Close']
        else:
            # fallback to first numeric column
            series = df.select_dtypes(include=["number"]).iloc[:, 0]

    df_prophet = pd.DataFrame({'ds': pd.to_datetime(df.index), 'y': series.values})

    m = Prophet()
    m.fit(df_prophet)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Prophet models are picklable via joblib
    import joblib
    joblib.dump(m, out_path)
    return out_path
