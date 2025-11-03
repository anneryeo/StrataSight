"""
Streamlit app for StrataSight - basic data exploration and model CRUD.
"""

import os
from pathlib import Path
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
    fig = px.line(df, x=df.index, y='Close', title=f'{selected} Close')
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
