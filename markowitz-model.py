import pandas as pd
import numpy  as np
import yfinance as yf
from docplex.mp.model import Model
import datetime

def get_stocks():
    tickers = [
        "AXP", "AAPL", "AMGN", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS",
        "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
        "PG", "TRV", "UNH", "V", "WBA", "WMT", "XOM"]

    # number of years
    n_years = 3.0

    # historical period
    end_date = datetime.datetime.today().date()
    start_date = end_date - datetime.timedelta(round(n_years * 365))

    print("\n=== DOWNLOAD HISTORICAL ASSET DATA ===")
    assets = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]

    assets.bfill(inplace=True)
    assets.ffill(inplace=True)

    return assets


def get_returns_data():

    daily_returns = get_stocks()
    daily_returns = daily_returns.diff()[1:] / daily_returns.shift(1)[1:]

    return  daily_returns

def report_stats():
    pass

def run_model():
    pass



if __name__ == "__main__":
    run_model()