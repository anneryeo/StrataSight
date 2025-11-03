"""Simple yfinance wrapper to fetch OHLCV for tickers."""
from typing import List
import yfinance as yf
import pandas as pd

DEFAULT_TICKERS = ["TSLA", "GME", "AMD", "AAPL"]


def fetch_tickers(tickers: List[str] = None, period: str = "1y", interval: str = "1d") -> dict:
    """Download data for a list of tickers using yfinance.

    Returns a dict mapping ticker -> DataFrame (Date index, columns Open/High/Low/Close/Volume)
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True)

    result = {}
    # If single ticker, yfinance returns single-level columns
    if len(tickers) == 1:
        result[tickers[0]] = data
        return result

    for t in tickers:
        try:
            df = data[t].copy()
            df.index = pd.to_datetime(df.index)
            result[t] = df
        except Exception:
            # fallback: try to download single ticker
            single = yf.download(t, period=period, interval=interval)
            single.index = pd.to_datetime(single.index)
            result[t] = single
    return result
