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
    # Prophet expects a DataFrame with ds and y
    df_prophet = df.reset_index()
    if 'Date' in df_prophet.columns:
        df_prophet = df_prophet.rename(columns={'Date': 'ds'})
    elif df_prophet.columns[0] != 'ds':
        df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds'})
    df_prophet = df_prophet[['ds', 'Close']].rename(columns={'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    m = Prophet()
    m.fit(df_prophet)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Prophet models are picklable via its internal utilities
    import joblib
    joblib.dump(m, out_path)
    return out_path
