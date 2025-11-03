"""
Streamlit app for StrataSight - basic data exploration and model CRUD.
"""

import os
import sys
from pathlib import Path
# Ensure project root is on sys.path so `import src.*` works when running script
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_fetch import fetch_tickers, DEFAULT_TICKERS
from src.models.lstm_model import train_and_save as train_lstm
from src.models.prophet_model import train_and_save as train_prophet

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='StrataSight', layout='wide')

st.title('StrataSight')

st.sidebar.header('Settings')
selected = st.sidebar.selectbox('Ticker', DEFAULT_TICKERS)
period = st.sidebar.selectbox('Period', ['1mo','3mo','6mo','1y','2y','5y'], index=3)

st.sidebar.markdown('---')
if st.sidebar.button('Fetch & Show'):
    data = fetch_tickers([selected], period=period)[selected]
    st.session_state['data'] = data

if 'data' in st.session_state:
    df = st.session_state['data']
    st.subheader(f'{selected} Close Price')

    # yfinance can return a DataFrame with MultiIndex columns when multiple
    # tickers are requested (e.g. ('TSLA','Close')). Detect that and
    # extract the appropriate Close series for the selected ticker before
    # plotting.
    try:
        import pandas as _pd

        if isinstance(df.columns, _pd.MultiIndex):
            close_key = (selected, 'Close')
            if close_key in df.columns:
                s = df[close_key]
            else:
                # fallback: try to find any column whose second level is 'Close'
                matches = [c for c in df.columns if c[1] == 'Close']
                s = df[matches[0]] if matches else df.iloc[:, 0]
        else:
            # single-level columns
            if 'Close' in df.columns:
                s = df['Close']
            else:
                # fallback to first numeric column
                s = df.iloc[:, 0]
    except Exception:
        # Very defensive fallback: pick first column/series
        s = df.iloc[:, 0]

    # Plot using the extracted series (plotly accepts x and y arrays)
    fig = px.line(x=s.index, y=s.values, labels={'x': 'Date', 'y': 'Close'}, title=f'{selected} Close')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Model actions')
    cols = st.columns(3)
    with cols[0]:
        if st.button('Train LSTM'):
            out = train_lstm(df, str(MODEL_DIR / f'{selected}_lstm'))
            st.success(f'LSTM saved to {out}')
    with cols[1]:
        if st.button('Train Prophet'):
            out = train_prophet(df, str(MODEL_DIR / f'{selected}_prophet.joblib'))
            st.success(f'Prophet saved to {out}')
    with cols[2]:
        st.markdown('Saved models:')
        files = list(MODEL_DIR.glob(f'{selected}_*'))
        for f in files:
            r = st.button(f'Delete {f.name}')
            if r:
                f.unlink()
                st.experimental_rerun()

else:
    st.info('Choose a ticker and click "Fetch & Show" in the sidebar to begin.')
