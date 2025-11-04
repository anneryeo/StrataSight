# StrataSight - Stock Price Prediction App with CRUD
# Streamlit frontend with LSTM and Prophet models for stock forecasting.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
from src.data_fetch import fetch_tickers, DEFAULT_TICKERS
from src.models.lstm_model import train_and_save as train_lstm
from src.models.prophet_model import train_and_save as train_prophet

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'saved'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# CRUD Database Path
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DB = DATA_DIR / 'predictions_history.json'

st.set_page_config(
    page_title='StrataSight',
    page_icon='',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============= CRUD OPERATIONS =============

def load_predictions_db():
    """READ: Load all predictions from JSON database"""
    if PREDICTIONS_DB.exists():
        with open(PREDICTIONS_DB, 'r') as f:
            return json.load(f)
    return []

def save_predictions_db(predictions):
    """Helper: Save predictions to JSON database"""
    with open(PREDICTIONS_DB, 'w') as f:
        json.dump(predictions, f, indent=2)

def create_prediction_record(ticker, forecast_days, current_price, lstm_pred, prophet_pred, notes=""):
    """CREATE: Add a new prediction record to database"""
    predictions = load_predictions_db()
    
    new_record = {
        'id': str(len(predictions) + 1),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ticker,
        'forecast_days': forecast_days,
        'current_price': float(current_price),
        'lstm_prediction': float(lstm_pred) if lstm_pred is not None else None,
        'prophet_prediction': float(prophet_pred) if prophet_pred is not None else None,
        'notes': notes
    }
    
    predictions.append(new_record)
    save_predictions_db(predictions)
    return new_record

def read_predictions():
    """READ: Get all predictions"""
    return load_predictions_db()

def update_prediction_notes(prediction_id, new_notes):
    """UPDATE: Modify notes for a specific prediction"""
    predictions = load_predictions_db()
    
    for pred in predictions:
        if pred['id'] == prediction_id:
            pred['notes'] = new_notes
            pred['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_predictions_db(predictions)
            return True
    return False

def delete_prediction(prediction_id):
    """DELETE: Remove a prediction record"""
    predictions = load_predictions_db()
    predictions = [p for p in predictions if p['id'] != prediction_id]
    save_predictions_db(predictions)
    return True

# =========================================

def get_or_train_lstm(ticker, forecast_days):
    model_path = MODEL_DIR / f'lstm_{ticker}.keras'
    
    df_dict = fetch_tickers([ticker])
    hist = df_dict[ticker].copy()
    
    # Handle MultiIndex columns (yfinance returns MultiIndex when fetching multiple tickers)
    if isinstance(hist.columns, pd.MultiIndex):
        # Extract the Close column - it will be (ticker, 'Close')
        close_col = [c for c in hist.columns if c[1] == 'Close'][0]
        close_series = hist[close_col]
        # Create a clean DataFrame with single-level columns
        hist = pd.DataFrame({'Close': close_series.values}, index=hist.index)
    
    # Train/load model
    if model_path.exists():
        st.info(f'Loading cached LSTM model for {ticker}...')
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))
    else:
        st.info(f'Training new LSTM model for {ticker}...')
        with st.spinner('Training LSTM model (this may take a few minutes)...'):
            train_lstm(hist, str(model_path), epochs=20, window=20)
        st.success(f'LSTM model trained and saved!')
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))
    
    # Generate predictions using the trained model
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(hist[['Close']].values)
    
    window = 20  # Must match the window used during training
    last_sequence = scaled[-window:]
    
    # Make predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(forecast_days):
        # Predict next value
        pred_scaled = model.predict(current_sequence.reshape(1, window, 1), verbose=0)[0, 0]
        predictions.append(pred_scaled)
        
        # Update sequence: remove oldest, add prediction
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Transform predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create forecast DataFrame
    last_date = hist.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    preds_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': predictions.flatten()}).set_index('Date')
    
    return model, preds_df

def get_or_train_prophet(ticker, forecast_days):
    import joblib
    model_path = MODEL_DIR / f'prophet_{ticker}.pkl'
    
    # Fetch historical data
    df_dict = fetch_tickers([ticker])
    hist = df_dict[ticker].copy()
    
    # Handle MultiIndex columns (yfinance returns MultiIndex when fetching multiple tickers)
    if isinstance(hist.columns, pd.MultiIndex):
        # Extract the Close column - it will be (ticker, 'Close')
        close_col = [c for c in hist.columns if c[1] == 'Close'][0]
        close_series = hist[close_col]
        # Create a clean DataFrame with single-level columns
        hist = pd.DataFrame({'Close': close_series.values}, index=hist.index)
    
    # Train or load model
    if model_path.exists():
        st.info(f'Loading cached Prophet model for {ticker}...')
        model = joblib.load(str(model_path))
    else:
        st.info(f'Training new Prophet model for {ticker}...')
        with st.spinner('Training Prophet model (this may take a minute)...'):
            train_prophet(hist, str(model_path))
        st.success(f'Prophet model trained and saved!')
        model = joblib.load(str(model_path))
    
    # Generate predictions
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Extract only future predictions
    last_date = hist.index[-1]
    preds_df = forecast[forecast['ds'] > last_date][['ds', 'yhat']].copy()
    preds_df.columns = ['Date', 'Predicted_Close']
    preds_df = preds_df.set_index('Date')
    
    return model, preds_df

def create_forecast_chart(historical_df, lstm_preds=None, prophet_preds=None, ticker=''):
    fig = go.Figure()
    
    # Handle MultiIndex columns
    if isinstance(historical_df.columns, pd.MultiIndex):
        close_col = [c for c in historical_df.columns if c[1] == 'Close'][0]
        close_data = historical_df[close_col]
    else:
        close_data = historical_df['Close']
    
    fig.add_trace(go.Scatter(x=historical_df.index, y=close_data, name='Historical', line=dict(color='#1f77b4', width=2), mode='lines'))
    if lstm_preds is not None and not lstm_preds.empty:
        fig.add_trace(go.Scatter(x=lstm_preds.index, y=lstm_preds['Predicted_Close'], name='LSTM Forecast', line=dict(color='#ff7f0e', width=2, dash='dash'), mode='lines+markers', marker=dict(size=6)))
    if prophet_preds is not None and not prophet_preds.empty:
        fig.add_trace(go.Scatter(x=prophet_preds.index, y=prophet_preds['Predicted_Close'], name='Prophet Forecast', line=dict(color='#2ca02c', width=2, dash='dot'), mode='lines+markers', marker=dict(size=6, symbol='square')))
    fig.update_layout(title=f'{ticker} Stock Price Forecast', xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified', template='plotly_white', height=600, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

def main():
    st.title('StrataSight')
    st.markdown('Get instant stock price forecasts using LSTM and Prophet models.')
    
    # Add navigation tabs
    tab1, tab2 = st.tabs(['Generate Predictions', 'Prediction History (CRUD)'])
    
    with tab1:
        prediction_tab()
    
    with tab2:
        history_tab()

def prediction_tab():
    """Main prediction generation interface"""
    
    # Initialize session state for persisting results
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    with st.sidebar:
        st.header('Prediction Settings')
        ticker = st.selectbox('Select Stock Ticker', options=DEFAULT_TICKERS, index=0, help='Choose the stock you want to predict')
        forecast_days = st.slider('Forecast Horizon (Days)', min_value=7, max_value=90, value=30, step=1, help='Number of days to forecast into the future')
        st.markdown('---')
        st.subheader('Select Models')
        use_lstm = st.checkbox('LSTM', value=True, help='Deep learning sequential model')
        use_prophet = st.checkbox('Prophet', value=True, help='Time series forecasting')
        st.markdown('---')
        generate_btn = st.button('Generate Predictions', type='primary', use_container_width=True)
        
        # Add clear results button if results exist
        if st.session_state.prediction_results is not None:
            if st.button('Clear Results', use_container_width=True):
                st.session_state.prediction_results = None
                st.rerun()
    
    # Check if we should generate new predictions
    if generate_btn:
        if not use_lstm and not use_prophet:
            st.error('Please select at least one model!')
            return
        
        # Display model information at the top
        with st.expander("About the Models", expanded=False):
            st.markdown("""
            **LSTM (Long Short-Term Memory)**
            - Deep learning neural network specialized for sequential data
            - Learns complex patterns from historical price movements
            - Better for capturing non-linear trends and short-term fluctuations
            - Can be sensitive to training data and may overfit
            
            **Prophet**
            - Facebook's time series forecasting tool
            - Designed for data with strong seasonal effects and trends
            - Robust to missing data and outliers
            - Better for stable trends and long-term forecasting
            - Handles holidays and special events well
            """)
        
        st.markdown('---')
        
        with st.spinner(f'Fetching historical data for {ticker}...'):
            df_dict = fetch_tickers([ticker])
            hist = df_dict[ticker].copy()
        
        # Handle MultiIndex columns
        if isinstance(hist.columns, pd.MultiIndex):
            close_col = [c for c in hist.columns if c[1] == 'Close'][0]
            close_series = hist[close_col]
            hist = pd.DataFrame({'Close': close_series.values}, index=hist.index)
        
        st.success(f'Loaded {len(hist)} days of historical data')
        lstm_preds = None
        prophet_preds = None
        if use_lstm:
            try:
                _, lstm_preds = get_or_train_lstm(ticker, forecast_days)
            except Exception as e:
                st.error(f'LSTM Error: {e}')
        if use_prophet:
            try:
                _, prophet_preds = get_or_train_prophet(ticker, forecast_days)
            except Exception as e:
                st.error(f'Prophet Error: {e}')
        
        # Store results in session state
        if lstm_preds is not None or prophet_preds is not None:
            st.session_state.prediction_results = {
                'ticker': ticker,
                'forecast_days': forecast_days,
                'hist': hist,
                'lstm_preds': lstm_preds,
                'prophet_preds': prophet_preds
            }
    
    # Display results (either just generated or from session state)
    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results
        ticker = results['ticker']
        forecast_days = results['forecast_days']
        hist = results['hist']
        lstm_preds = results['lstm_preds']
        prophet_preds = results['prophet_preds']
        
        st.markdown('---')
        st.subheader('Forecast Results')
        col1, col2, col3 = st.columns(3)
        
        lstm_final = None
        prophet_final = None
        
        with col1:
            current_price = hist['Close'].iloc[-1]
            st.metric('Current Price', f'${current_price:.2f}')
        with col2:
            if lstm_preds is not None:
                lstm_final = lstm_preds['Predicted_Close'].iloc[-1]
                lstm_change = ((lstm_final - current_price) / current_price) * 100
                st.metric('LSTM Prediction', f'${lstm_final:.2f}', f'{lstm_change:+.2f}%')
        with col3:
            if prophet_preds is not None:
                prophet_final = prophet_preds['Predicted_Close'].iloc[-1]
                prophet_change = ((prophet_final - current_price) / current_price) * 100
                st.metric('Prophet Prediction', f'${prophet_final:.2f}', f'{prophet_change:+.2f}%')
        
        # Add Save Prediction Button (CREATE operation)
        st.markdown('---')
        with st.expander('Save This Prediction', expanded=False):
            save_notes = st.text_area('Add notes (optional):', placeholder='e.g., Strong upward trend expected...', key='save_notes_input')
            if st.button('Save Prediction to History', type='primary', key='save_pred_btn'):
                lstm_val = lstm_final if lstm_preds is not None else None
                prophet_val = prophet_final if prophet_preds is not None else None
                
                record = create_prediction_record(
                    ticker=ticker,
                    forecast_days=forecast_days,
                    current_price=current_price,
                    lstm_pred=lstm_val,
                    prophet_pred=prophet_val,
                    notes=save_notes
                )
                st.success(f'Prediction saved! (ID: {record["id"]})')
                st.balloons()
        
        st.plotly_chart(create_forecast_chart(hist, lstm_preds, prophet_preds, ticker), use_container_width=True)
        st.markdown('---')
        st.subheader('Detailed Predictions')
        tab_cols = []
        if lstm_preds is not None:
            tab_cols.append('LSTM')
        if prophet_preds is not None:
            tab_cols.append('Prophet')
        tabs = st.tabs(tab_cols)
        tab_idx = 0
        if lstm_preds is not None:
            with tabs[tab_idx]:
                st.dataframe(lstm_preds, use_container_width=True)
            tab_idx += 1
        if prophet_preds is not None:
            with tabs[tab_idx]:
                st.dataframe(prophet_preds, use_container_width=True)
    else:
        # Show welcome screen when no results
        st.info('Configure your settings in the sidebar and click Generate Predictions to start!')
        st.markdown('### How to Use StrataSight\n\n1. **Select a Stock**: Choose from TSLA, GME, AMD, or AAPL\n2. **Set Forecast Horizon**: Pick how many days ahead you want to predict (7-90 days)\n3. **Choose Models**: Select LSTM, Prophet, or both\n4. **Generate**: Click the button to get instant predictions!\n\n### About the Models\n\n- **LSTM**: A deep learning model that learns patterns from historical price sequences\n- **Prophet**: Time series forecasting tool designed for strong seasonal effects\n\nThe first prediction may take a few minutes as the model trains. Subsequent predictions are instant!')
        st.markdown('---')
        st.subheader('Recent Historical Data')
        ticker_preview = st.selectbox('Preview ticker:', DEFAULT_TICKERS, key='preview_ticker')
        with st.spinner('Fetching data...'):
            df_dict = fetch_tickers([ticker_preview])
            hist_preview = df_dict[ticker_preview]
            # Handle MultiIndex columns for display
            if isinstance(hist_preview.columns, pd.MultiIndex):
                close_col = [c for c in hist_preview.columns if c[1] == 'Close'][0]
                close_series = hist_preview[close_col]
                hist_display = pd.DataFrame({'Close': close_series.values}, index=hist_preview.index)
            else:
                hist_display = hist_preview
        st.dataframe(hist_display.tail(10), use_container_width=True)

def history_tab():
    """CRUD interface for prediction history"""
    st.subheader('Prediction History Database')
    st.markdown('Manage your saved predictions with full CRUD operations (Create, Read, Update, Delete)')
    
    # READ: Load all predictions
    predictions = read_predictions()
    
    if not predictions:
        st.info('No saved predictions yet. Generate and save predictions from the "Generate Predictions" tab!')
        return
    
    st.success(f'Found {len(predictions)} saved prediction(s)')
    
    # Display predictions in a table
    st.markdown('---')
    st.subheader('All Saved Predictions (READ)')
    
    # Convert to DataFrame for display
    df_display = pd.DataFrame(predictions)
    display_cols = ['id', 'timestamp', 'ticker', 'forecast_days', 'current_price', 
                    'lstm_prediction', 'prophet_prediction', 'notes']
    df_display = df_display[display_cols]
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # UPDATE and DELETE operations
    st.markdown('---')
    st.subheader('Update or Delete Predictions')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### UPDATE: Edit Notes')
        prediction_ids = [p['id'] for p in predictions]
        selected_id_update = st.selectbox('Select Prediction ID to Update:', prediction_ids, key='update_select')
        
        # Find selected prediction
        selected_pred = next((p for p in predictions if p['id'] == selected_id_update), None)
        
        if selected_pred:
            st.info(f"**Current Notes:** {selected_pred.get('notes', 'No notes')}")
            new_notes = st.text_area('Enter new notes:', value=selected_pred.get('notes', ''), key='new_notes')
            
            if st.button('Update Notes', type='primary', key='update_btn'):
                if update_prediction_notes(selected_id_update, new_notes):
                    st.success(f'Prediction ID {selected_id_update} updated successfully!')
                    st.rerun()
                else:
                    st.error('Update failed!')
    
    with col2:
        st.markdown('### DELETE: Remove Prediction')
        selected_id_delete = st.selectbox('Select Prediction ID to Delete:', prediction_ids, key='delete_select')
        
        # Find selected prediction for preview
        selected_pred_del = next((p for p in predictions if p['id'] == selected_id_delete), None)
        
        if selected_pred_del:
            st.warning(f"**Ticker:** {selected_pred_del['ticker']} | **Date:** {selected_pred_del['timestamp']}")
            
            if st.button('Delete Prediction', type='secondary', key='delete_btn'):
                if delete_prediction(selected_id_delete):
                    st.success(f'Prediction ID {selected_id_delete} deleted!')
                    st.rerun()
                else:
                    st.error('Deletion failed!')
    
    # Bulk operations
    st.markdown('---')
    st.subheader('Bulk Operations')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Export All Predictions (JSON)', use_container_width=True):
            json_str = json.dumps(predictions, indent=2)
            st.download_button(
                label='Download JSON',
                data=json_str,
                file_name=f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )
    
    with col2:
        if st.button('Delete All Predictions', use_container_width=True):
            save_predictions_db([])
            st.success('All predictions deleted!')
            st.rerun()

if __name__ == '__main__':
    main()
