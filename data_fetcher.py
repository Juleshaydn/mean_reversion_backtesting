import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, period='1mo', interval='1d'):
    """
    Fetches historical stock data from yfinance.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    hist['symbol'] = symbol
    # Ensure the 'Date' or 'Datetime' column exists
    date_column = 'Date' if 'Date' in hist.columns else 'Datetime'
    hist.rename(columns={date_column: 'date'}, inplace=True)
    hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)  # Remove timezone
    return hist
