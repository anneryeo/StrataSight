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

    # Buttons: Train & Forecast and Show saved models
    cols = st.columns((1,1,1))
    with cols[0]:
        if st.button('Train & Forecast'):
            # perform training (or load) and generate forecasts for selected models
            forecasts = {}
            # extract the series again (reuse s)
            history = s
            # LSTM
            if 'LSTM' in models_to_run:
                lstm_path = MODEL_DIR / f'{selected}_lstm.keras'
                # if saved model exists and allowed, load; otherwise train
                if lstm_path.exists() and use_saved_only:
                    import tensorflow as _tf
                    model = _tf.keras.models.load_model(str(lstm_path))
                else:
                    train_and = train_lstm(df, str(MODEL_DIR / f'{selected}_lstm'))
                    import tensorflow as _tf
                    model = _tf.keras.models.load_model(train_and)

                # iterative forecasting using last window
                window = 20
                last_vals = history.values[-window:].astype('float32')
                preds = []
                import numpy as _np
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
                    train_out = train_prophet(df, str(prop_path))
                    m = _joblib.load(train_out)

                future = m.make_future_dataframe(periods=forecast_days, freq='D')
                forecast = m.predict(future)
                # get the forecast tail corresponding to future periods
                f_dates = pd.to_datetime(forecast['ds'].tail(forecast_days)).values
                f_vals = forecast['yhat'].tail(forecast_days).values
                forecasts['Prophet'] = pd.Series(data=f_vals, index=pd.to_datetime(f_dates))

            # Combine history and forecasts into a single DataFrame for plotting
            plot_df = pd.DataFrame({'Date': history.index, 'History': history.values})
            plot_df = plot_df.set_index('Date')
            for name, series in forecasts.items():
                plot_df = plot_df.join(series.rename(name), how='outer')

            # Long format for plotly express
            plot_long = plot_df.reset_index().melt(id_vars='Date', var_name='Series', value_name='Close')
            fig2 = px.line(plot_long, x='Date', y='Close', color='Series', title=f'{selected} - Historical and Forecasts', labels={'Close':'Price', 'Date':'Date'})
            fig2.update_layout(legend_title_text='Series')
            st.plotly_chart(fig2, use_container_width=True)
            # show persisted model files
            st.success('Training and forecasting complete.')
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
