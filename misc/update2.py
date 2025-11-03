import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

pip install prophet

def get_data(ticker="AAPL", start="2016-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    return df

def prepare_data(df, seq_len=60, feature_col="Close"):
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def split_data(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test

def build_lstm(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_lstm(X_train, y_train, X_test, y_test,
               epochs=50, batch_size=32, model_path="best_lstm_model.h5"):
    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Best model saved at: {model_path}")
    return model, history

if __name__ == "__main__":
    df = get_data("AAPL", start="2016-01-01")
    print(f"Downloaded {len(df)} rows of data.")
    X, y, scaler = prepare_data(df, seq_len=60)
    X_train, y_train, X_test, y_test = split_data(X, y)
    model, history = train_lstm(X_train, y_train, X_test, y_test, epochs=30)
    predictions = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    preds_inv = scaler.inverse_transform(predictions)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

st.title("📊 Stock Price Forecasting Dashboard")
st.sidebar.header("Configuration")
st.sidebar.header("Stock Selection")
ticker = st.sidebar.selectbox(
    "Choose a Stock:",
    options=["AAPL", "AMD", "TSLA"],
    format_func=lambda x: {"AAPL": "Apple", "AMD": "AMD", "TSLA": "Tesla"}[x]
)
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 7)
data = yf.download(ticker, start="2016-01-01", progress=False)
st.subheader(f"Stock Data for {ticker}")
st.dataframe(data.tail())
ml_selection = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Long Short-Term Memory (LSTM)", "Prophet"]
)
if ml_selection == "Long Short-Term Memory (LSTM)":
    st.subheader(f"LSTM Forecast for {ticker}")
    with st.spinner("Training LSTM model..."):
        seq_len = 60
        X_train, y_train, X_test, y_test, scaler = prepare_data(data, seq_len)
        model, history = train_lstm(X_train, y_train, X_test, y_test, epochs=20)
        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        result_df = pd.DataFrame({
            "Date": data.index[-len(preds):],
            "Actual": actual.flatten(),
            "Predicted": preds.flatten()
        })
    st.subheader("Forecast Data")
    st.dataframe(result_df.tail())
    st.subheader("Forecast Plot")
    st.line_chart(result_df.set_index("Date")[["Actual", "Predicted"]])
elif ml_selection == "Prophet":
    st.subheader(f"Prophet Forecast for {ticker}")
    df = data.reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    with st.spinner("Training Prophet model..."):
        model = Prophet(daily_seasonality=True)
        model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    st.subheader("Forecast Data")
    st.dataframe(forecast.tail())
    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)
