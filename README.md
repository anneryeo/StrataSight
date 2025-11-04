# StrataSight

StrataSight is a stock price forecasting application with full CRUD functionality that uses machine learning models to predict future stock prices. Built with Streamlit for an intuitive interface, it leverages yfinance for real-time market data, TensorFlow/Keras LSTM and Facebook's Prophet for forecasting, and Plotly for interactive visualizations.

## Features

### Prediction Generation
- **Simple One-Click Predictions**: Select a stock, choose forecast horizon, pick models, and generate predictions instantly
- **Dual Model Approach**:
  - **LSTM (Long Short-Term Memory)**: Deep learning neural network for capturing complex patterns and short-term fluctuations
  - **Prophet**: Facebook's robust time series forecasting tool for stable trends and seasonal effects
- **Interactive Visualizations**: Plotly charts showing historical data and model predictions side-by-side
- **Smart Model Caching**: Models train once per stock and are cached for instant subsequent predictions
- **Comparison Metrics**: View predicted prices with percentage changes from current values
- **Session State Persistence**: Results remain visible across page interactions

### CRUD Operations (Prediction History Management)
- **CREATE**: Save prediction results with optional notes to JSON database
- **READ**: View all saved predictions in a searchable table format
- **UPDATE**: Edit notes on existing predictions with timestamp tracking
- **DELETE**: Remove individual predictions or clear entire database
- **EXPORT**: Download prediction history as JSON for backup/analysis
- **Database**: File-based JSON storage in `data/predictions_history.json`

## Quick Setup (Windows PowerShell)

1. Create and activate a virtual environment

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2. Install dependencies

    ```powershell
    pip install -r requirements.txt
    ```

3. Run the Streamlit app locally

    ```powershell
    streamlit run src\stratasight.py
    ```

    Or, to use the virtual environment's Python explicitly:

    ```powershell
    & ".\.venv\Scripts\python.exe" -m streamlit run src\stratasight.py
    ```

The app will open in your browser at `http://localhost:8501`

## How to Use

### Generate Predictions Tab
1. **Select a Stock**: Choose from TSLA, GME, AMD, or AAPL
2. **Set Forecast Horizon**: Pick how many days ahead to predict (7-90 days)
3. **Choose Models**: Select LSTM, Prophet, or both for comparison
4. **Generate Predictions**: Click the button and view results with interactive charts
5. **Save Results** (optional): Expand "Save This Prediction" to store results with notes

### Prediction History (CRUD) Tab
- **View History**: Browse all saved predictions with timestamps and details
- **Update Notes**: Select a prediction ID and edit its notes
- **Delete Records**: Remove individual predictions or clear all at once
- **Export Data**: Download prediction history as JSON backup

## Technical Notes

- **Prophet Installation**: Can be system-dependent. See [Prophet documentation](https://facebook.github.io/prophet/docs/installation.html) for OS-specific requirements
- **Model Storage**: Trained models are cached in `models/saved/` directory
  - LSTM models: `.keras` format
  - Prophet models: `.pkl` format (joblib)
- **Data Storage**: Prediction history saved in `data/predictions_history.json`
  - JSON format for human-readable storage
  - Includes ID, timestamp, ticker, prices, predictions, and notes
  - No external database required
- **Data Normalization**: LSTM uses MinMaxScaler for consistent training and prediction
- **Session State**: Streamlit session state preserves predictions across page interactions
- **Architecture**: 
  - LSTM: 2-layer stacked LSTM with dropout regularization
  - Prophet: Handles trends, seasonality, and outliers automatically

## Deployment

For deployment to Streamlit Cloud:
1. Push your repository to GitHub
2. Connect to Streamlit Cloud
3. Point the app to `src/stratasight.py`
4. The included `requirements.txt` will handle dependencies

## Project Structure

```
StrataSight/
├── src/
│   ├── stratasight.py           # Main Streamlit app with CRUD
│   ├── data_fetch.py             # yfinance data fetching
│   └── models/
│       ├── lstm_model.py         # LSTM training and inference
│       └── prophet_model.py      # Prophet training and inference
├── models/
│   └── saved/                    # Cached trained models (.keras, .pkl)
├── data/
│   └── predictions_history.json  # CRUD database (auto-created)
├── requirements.txt              # Python dependencies
├── CRUD_DOCUMENTATION.md         # Technical CRUD documentation
├── CRUD_USER_GUIDE.md            # User guide for CRUD features
└── README.md
```
