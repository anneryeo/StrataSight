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
# JOBS is persisted into session_state and to disk so refreshes don't hide
# running/finished jobs. THREADS holds in-memory Thread objects for the
# currently active threads (not persisted across reruns).
THREADS = {}





MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Persistence paths (jobs metadata, progress log, and full job results)
JOBS_FILE = MODEL_DIR.parent / 'jobs.json'
JOB_PROGRESS_FILE = MODEL_DIR.parent / 'job_progress.log'
JOB_RESULTS_DIR = MODEL_DIR.parent / 'job_results'
JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_jobs_from_disk():
    import json
    try:
        if JOBS_FILE.exists():
            with open(JOBS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _persist_jobs_to_disk():
    import json
    try:
        serial = {}
        for k, v in JOBS.items():
            serial[k] = {
                'status': v.get('status'),
                'started': v.get('started'),
                'finished': v.get('finished'),
                'progress': v.get('progress'),
                'error': v.get('error'),
                'result_path': str(v.get('result_path')) if v.get('result_path') is not None else None,
            }
        with open(JOBS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serial, f, indent=2)
    except Exception:
        try:
            _log('failed to persist jobs to disk')
        except Exception:
            pass


st.set_page_config(page_title='StrataSight', layout='wide')

st.title('StrataSight')

# Initialize JOBS in session state so it survives script reruns in the same
# browser session. If persisted jobs exist on disk, load them.
if 'JOBS' not in st.session_state:
    st.session_state['JOBS'] = _load_jobs_from_disk()

# Alias to a local name used throughout the file. This is a live dict and
# background threads will update it in place; we persist changes to disk.
JOBS = st.session_state['JOBS']


def safe_rerun():
    """Attempt to programmatically rerun the Streamlit script.

    Uses st.experimental_rerun() when available. If not present (older/newer
    runtime variations), attempts to raise the internal RerunException. If
    that also fails, sets a session flag and calls st.stop() so the UI does a
    natural rerun on next interaction.
    """
    rerun = getattr(st, 'experimental_rerun', None)
    if callable(rerun):
        try:
            rerun()
            return
        except Exception:
            pass
    # If experimental rerun isn't available, set a session flag and stop
    try:
        st.session_state['_needs_refresh'] = True
    except Exception:
        pass
    st.stop()


def _log(msg: str):
    """Simple stdout logger (visible in streamlit.log)."""
    try:
        print(f"[STRATASIGHT] {msg}", flush=True)
    except Exception:
        pass


st.sidebar.header('Settings')
selected = st.sidebar.selectbox('Ticker', DEFAULT_TICKERS)
period = st.sidebar.selectbox('Period', ['1mo','3mo','6mo','1y','2y','5y'], index=3)

st.sidebar.markdown('')
st.sidebar.caption('Fetch and show the selected ticker for the chosen period')
if st.sidebar.button('Fetch & Show current data (selected ticker and period)'): 
    data = fetch_tickers([selected], period=period)[selected]
    st.session_state['data'] = data

# Forecast controls
st.sidebar.markdown('---')
forecast_days = st.sidebar.number_input('Forecast days', min_value=1, max_value=365, value=30)
models_to_run = st.sidebar.multiselect('Models to run', ['LSTM', 'Prophet'], default=['LSTM','Prophet'])
use_saved_only = st.sidebar.checkbox('Use saved models if available', value=True)
auto_refresh = st.sidebar.checkbox('Auto-refresh job status', value=True)
refresh_interval = st.sidebar.number_input('Refresh interval (s)', min_value=1, max_value=60, value=2)

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


    def generate_forecasts(job_name, df, selected, models_to_run, forecast_days, use_saved_only):
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
                # when training in background, provide a progress callback that
                # writes back to JOBS[job_name]['progress'] so the UI can show a progress bar
                def _progress_cb(frac):
                    try:
                        JOBS[job_name]['progress'] = float(frac)
                        # persist metadata and write a progress line for external
                        # inspection
                        _persist_jobs_to_disk()
                    except Exception:
                        pass
                    # Log progress so we can see it in streamlit.log and also
                    # append to a small progress file for external inspection.
                    try:
                        _log(f"progress update: job={job_name} frac={frac}")
                        with open(JOB_PROGRESS_FILE, 'a', encoding='utf-8') as pf:
                            pf.write(f"{time.time()} {job_name} {frac}\n")
                    except Exception:
                        pass

                trained = train_lstm(df, str(MODEL_DIR / f'{selected}_lstm'), progress_callback=_progress_cb)
                model = _tf.keras.models.load_model(trained)

            window = 20
            last_vals = history.values[-window:].astype('float32')
            preds = []
            for _ in range(forecast_days):
                x = _np.array(last_vals).reshape((1, window, 1))
                yhat = model.predict(x, verbose=0)[0,0]
                preds.append(float(yhat))
                last_vals = _np.append(last_vals[1:], yhat)

            dates = pd.date_range(start=history.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
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

            future = m.make_future_dataframe(periods=forecast_days, freq='B')
            forecast = m.predict(future)
            f_dates = pd.to_datetime(forecast['ds'].tail(forecast_days)).values
            f_vals = forecast['yhat'].tail(forecast_days).values
            forecasts['Prophet'] = pd.Series(data=f_vals, index=pd.to_datetime(f_dates))

        return history, forecasts


    def _background_target(job_name, df, selected, models_to_run, forecast_days, use_saved_only):
        try:
            _log(f"starting job {job_name}")
            JOBS[job_name] = {'status': 'running', 'started': time.time(), 'progress': 0.0}
            _persist_jobs_to_disk()
            # write a coarse start marker to the progress log
            try:
                with open(JOB_PROGRESS_FILE, 'a', encoding='utf-8') as pf:
                    pf.write(f"{time.time()} {job_name} START\n")
            except Exception:
                pass

            history, forecasts = generate_forecasts(job_name, df, selected, models_to_run, forecast_days, use_saved_only)

            # Save full result object to disk with joblib so it can be loaded
            # across script reruns.
            try:
                import joblib as _joblib
                result_path = JOB_RESULTS_DIR / f"{job_name}_result.pkl"
                _joblib.dump({'history': history, 'forecasts': forecasts}, str(result_path))
                JOBS[job_name].update({'status': 'done', 'result_path': str(result_path), 'finished': time.time(), 'progress': 1.0})
            except Exception as e:
                JOBS[job_name].update({'status': 'error', 'error': str(e)})
                _log(f"job {job_name} result save error: {e}")

            _persist_jobs_to_disk()
            _log(f"finished job {job_name}")
        except Exception as e:
            JOBS[job_name].update({'status': 'error', 'error': str(e)})
            _persist_jobs_to_disk()
            _log(f"job {job_name} error: {e}")


    def start_job(job_name, df, selected, models_to_run, forecast_days, use_saved_only):
        if job_name in JOBS and JOBS[job_name].get('status') == 'running':
            return False
        t = threading.Thread(target=_background_target, args=(job_name, df, selected, models_to_run, forecast_days, use_saved_only), daemon=True)
        JOBS[job_name] = {'status': 'queued', 'started': time.time(), 'progress': 0.0}
        THREADS[job_name] = t
        _persist_jobs_to_disk()
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
                    safe_rerun()
    with cols[2]:
        st.markdown('Model help')
        st.write('Saved models are reusable artifacts — you can reuse them later instead of retraining. Use the sidebar checkbox "Use saved models if available" to prefer loading them.')

    # Job status panel
    st.markdown('### Job status')
    if JOBS:
        if st.button('Refresh job status'):
            safe_rerun()
        # Debug viewers: show the persisted jobs.json or tail the progress log
        if st.button('Show raw jobs.json'):
            try:
                import json
                if JOBS_FILE.exists():
                    with open(JOBS_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    st.code(json.dumps(data, indent=2))
                else:
                    st.json(JOBS)
            except Exception as e:
                st.error(f'Failed to load jobs.json: {e}')

        if st.button('Tail job_progress.log'):
            try:
                if JOB_PROGRESS_FILE.exists():
                    with open(JOB_PROGRESS_FILE, 'r', encoding='utf-8') as pf:
                        lines = pf.readlines()[-200:]
                    st.code(''.join(lines))
                else:
                    st.write('No job_progress.log found yet.')
            except Exception as e:
                st.error(f'Failed to read job progress log: {e}')
        for jname, info in list(JOBS.items()):
            status = info.get('status', 'unknown')
            started = info.get('started')
            finished = info.get('finished')
            col1, col2 = st.columns((3,1))
            col1.write(f"Job: **{jname}** — Status: **{status}**")
            if status == 'running' and started:
                elapsed = int(time.time() - started)
                col2.info(f"Running for {elapsed}s")
                # show progress bar if available
                prog = info.get('progress')
                if prog is not None:
                    try:
                        p = int(min(max(float(prog) * 100.0, 0.0), 100.0))
                        col2.progress(p)
                    except Exception:
                        pass
            elif status == 'queued':
                col2.info('Queued')
            elif status == 'done':
                col2.success('Done')
                # if result not yet copied into session, offer to load
                if 'forecasts' not in st.session_state or st.session_state.get('loaded_job') != jname:
                        if col2.button(f'Load results ({jname})'):
                            res_path = info.get('result_path')
                            if res_path:
                                try:
                                    import joblib as _joblib
                                    res = _joblib.load(res_path)
                                    st.session_state['history'] = res.get('history')
                                    st.session_state['forecasts'] = res.get('forecasts')
                                    st.session_state['loaded_job'] = jname
                                    safe_rerun()
                                except Exception as e:
                                    st.error(f'Failed to load results: {e}')
                else:
                    col2.write('Loaded')
            elif status == 'error':
                col2.error('Error')
                col1.write(f"**Error:** {info.get('error')}" )
    else:
        st.write('No jobs yet. Start a Train & Forecast or Run Forecast job.')

    # Auto-refresh polling: if enabled and any job is running, poll and rerun
    if auto_refresh and any(info.get('status') == 'running' for info in JOBS.values()):
        time.sleep(int(refresh_interval))
        safe_rerun()

    # If forecasts are stored in session, show the combined plot (persistent across refreshes)
    if 'forecasts' in st.session_state and 'history' in st.session_state:
        hist = st.session_state['history']
        forecasts = st.session_state['forecasts']
        # Build a DataFrame with a proper datetime index and a named index
        # so reset_index() reliably produces a 'Date' column. Use the
        # history Series values as the base and join forecast Series by
        # index (dates).
        try:
            idx = pd.to_datetime(hist.index)
        except Exception:
            idx = pd.to_datetime(pd.Index(hist.index))
        plot_df = pd.DataFrame({'History': hist.values}, index=idx)
        plot_df.index.name = 'Date'
        for name, series in forecasts.items():
            # ensure forecast index is datetime
            try:
                fidx = pd.to_datetime(series.index)
                s = pd.Series(series.values, index=fidx, name=name)
            except Exception:
                s = pd.Series(series.values, index=series.index, name=name)
            plot_df = plot_df.join(s, how='outer')
        
        # Ensure index is named 'Date' before reset_index (join may clear the name)
        if plot_df.index.name != 'Date':
            plot_df.index.name = 'Date'
        
        plot_long = plot_df.reset_index().melt(id_vars='Date', var_name='Series', value_name='Close')
        fig_saved = px.line(plot_long, x='Date', y='Close', color='Series', title=f'{selected} - Historical and Forecasts (loaded)', labels={'Close':'Price', 'Date':'Date'})
        fig_saved.update_layout(legend_title_text='Series')
        st.plotly_chart(fig_saved, use_container_width=True)

else:
    st.info('Choose a ticker and click "Fetch & Show" in the sidebar to begin.')
