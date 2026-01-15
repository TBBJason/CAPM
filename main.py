import yfinance as yf
import pandas as pd
import numpy as np


def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']


if __name__ == "__main__":
    # data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
    # print(data.head())
    aapl = download_stock_data(("AAPL", "MSFT"), '2020-01-01', '2021-01-01')
    print(aapl.head())


