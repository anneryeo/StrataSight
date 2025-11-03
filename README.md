# StrataSight

StrataSight is a small CRUD-style stock model exploration app using Streamlit for the frontend, yfinance as the data source, TensorFlow/Keras LSTM and Prophet for models, and Plotly/Matplotlib for visualizations.

Quick setup (Windows PowerShell):

1. Create and activate a virtual environment

    python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies

    pip install -r requirements.txt

3. Run the Streamlit app locally

    streamlit run src\stratasight.py

Notes:
- Prophet installation can be system-dependent. See documentation for Prophet to check for OS-specific stuff.
- For deployment to Streamlit Cloud, create a `requirements.txt` (already included) and point the app to `src/stratasight.py`.
