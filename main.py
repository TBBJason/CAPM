import yfinance as yf
import pandas as pd
import numpy as np
from portfolio import tangency_weights, tangency_weights_constrained, efficient_frontier


# downloading and loading data
def download_stock_data(tickers, start, end):
    """Download adjusted close prices, always returning a DataFrame.

    yfinance returns a Series for a single ticker; we normalise to a one-column
    DataFrame so callers can rely on a consistent shape and on ``.columns``.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True)
    close = data['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def calculate_annualized_return(returns, periods_per_year=252):
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1


def calculate_mu_sigma(returns, annualized=True, periods_per_year=252):
    mu = returns.mean()
    sigma = returns.cov()
    if annualized:
        mu = mu * periods_per_year
        sigma = sigma * periods_per_year
    return mu.values, sigma.values


if __name__ == "__main__":
    stocks = download_stock_data(("AAPL", "MSFT"), '2023-01-01', '2024-01-01')
    print(stocks.head())
