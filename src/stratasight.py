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
import threading
import time

# simple in-memory job store for background tasks
JOBS = {}

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='StrataSight', layout='wide')

st.title('StrataSight')

st.sidebar.header('Settings')
selected = st.sidebar.selectbox('Ticker', DEFAULT_TICKERS)
period = st.sidebar.selectbox('Period', ['1mo','3mo','6mo','1y','2y','5y'], index=3)

st.sidebar.markdown('')
st.sidebar.caption('Fetch and show the selected ticker for the chosen period')
if st.sidebar.button('Fetch & Show (selected ticker and period)'):
    data = fetch_tickers([selected], period=period)[selected]
    st.session_state['data'] = data

# Forecast controls
st.sidebar.markdown('---')
forecast_days = st.sidebar.number_input('Forecast days', min_value=1, max_value=365, value=30)
models_to_run = st.sidebar.multiselect('Models to run', ['LSTM', 'Prophet'], default=['LSTM','Prophet'])
use_saved_only = st.sidebar.checkbox('Use saved models if available', value=True)

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

    # Show small help text about model actions
    st.info('Train models to generate forecasts. Select models and forecast days in the sidebar, then click "Train & Forecast".')


    def generate_forecasts(df, selected, models_to_run, forecast_days, use_saved_only):
        """Generate forecasts for selected models. Returns dict[name->Series]."""
        forecasts = {}
        # extract history series (reuse earlier logic)
        try:
            if isinstance(df.columns, pd.MultiIndex):
                close_key = (selected, 'Close')
                if close_key in df.columns:
                    history = df[close_key]
                else:
                    matches = [c for c in df.columns if c[1] == 'Close']
                    history = df[matches[0]] if matches else df.iloc[:, 0]
            else:
                history = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        except Exception:
            history = df.iloc[:, 0]

        # LSTM
        if 'LSTM' in models_to_run:
            lstm_path = MODEL_DIR / f'{selected}_lstm.keras'
            import numpy as _np
            import tensorflow as _tf
            if lstm_path.exists() and use_saved_only:
                model = _tf.keras.models.load_model(str(lstm_path))
            else:
                trained = train_lstm(df, str(MODEL_DIR / f'{selected}_lstm'))
                model = _tf.keras.models.load_model(trained)

            window = 20
            last_vals = history.values[-window:].astype('float32')
            preds = []
            for _ in range(forecast_days):
                x = _np.array(last_vals).reshape((1, window, 1))
                yhat = model.predict(x, verbose=0)[0,0]
                preds.append(float(yhat))
                last_vals = _np.append(last_vals[1:], yhat)

            dates = pd.date_range(start=history.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            forecasts['LSTM'] = pd.Series(data=preds, index=dates)

        # Prophet
        if 'Prophet' in models_to_run:
            prop_path = MODEL_DIR / f'{selected}_prophet.joblib'
            import joblib as _joblib
            if prop_path.exists() and use_saved_only:
                m = _joblib.load(str(prop_path))
            else:
                trained = train_prophet(df, str(prop_path))
                m = _joblib.load(trained)

            future = m.make_future_dataframe(periods=forecast_days, freq='D')
            forecast = m.predict(future)
            f_dates = pd.to_datetime(forecast['ds'].tail(forecast_days)).values
            f_vals = forecast['yhat'].tail(forecast_days).values
            forecasts['Prophet'] = pd.Series(data=f_vals, index=pd.to_datetime(f_dates))

        return history, forecasts


    def _background_target(job_name, df, selected, models_to_run, forecast_days, use_saved_only):
        try:
            JOBS[job_name] = {'status': 'running', 'started': time.time()}
            history, forecasts = generate_forecasts(df, selected, models_to_run, forecast_days, use_saved_only)
            JOBS[job_name].update({'status': 'done', 'result': {'history': history, 'forecasts': forecasts}, 'finished': time.time()})
        except Exception as e:
            JOBS[job_name].update({'status': 'error', 'error': str(e)})


    def start_job(job_name, df, selected, models_to_run, forecast_days, use_saved_only):
        if job_name in JOBS and JOBS[job_name].get('status') == 'running':
            return False
        t = threading.Thread(target=_background_target, args=(job_name, df, selected, models_to_run, forecast_days, use_saved_only), daemon=True)
        JOBS[job_name] = {'status': 'queued', 'thread': t}
        t.start()
        return True


    # Buttons: Train & Forecast and Show saved models
    cols = st.columns((1,1,1))
    with cols[0]:
        if st.button('Train & Forecast'):
            # start a background job to train (or load) and forecast
            started = start_job('train_forecast', df, selected, models_to_run, forecast_days, use_saved_only)
            if not started:
                st.warning('A training job is already running.')
            else:
                st.info('Training job started in the background. Refresh or check job status below.')
        if st.button('Run Forecast'):
            started = start_job('run_forecast', df, selected, models_to_run, forecast_days, use_saved_only)
            if not started:
                st.warning('A forecast job is already running.')
            else:
                st.info('Forecast job started in the background. Refresh or check job status below.')
    with cols[1]:
        if st.button('Show saved models'):
            st.markdown('Saved models for the selected ticker:')
            files = list(MODEL_DIR.glob(f'{selected}_*'))
            if not files:
                st.write('No saved models found.')
            for f in files:
                c1, c2 = st.columns((3,1))
                c1.write(f.name)
                if c2.button(f'Load & Predict ({f.name})'):
                    st.info(f'Loaded {f.name} - use Train & Forecast to regenerate predictions or re-run with models selected.')
                if c2.button(f'Delete {f.name}'):
                    f.unlink()
                    st.experimental_rerun()
    with cols[2]:
        st.markdown('Model help')
        st.write('Saved models are reusable artifacts — you can reuse them later instead of retraining. Use the sidebar checkbox "Use saved models if available" to prefer loading them.')

else:
    st.info('Choose a ticker and click "Fetch & Show" in the sidebar to begin.')
