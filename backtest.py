from main import download_stock_data
import pandas as pd
import numpy as np


def backtest(stocks, weights, prices=None, start=None, end=None):
    """Compute the cumulative value of a portfolio over a period.

    Parameters
    ----------
    stocks : sequence of str
        Tickers, used only when ``prices`` is not supplied.
    weights : array-like
        Portfolio weights. Must align with the *column order* of ``prices``
        (i.e. the order returned by yfinance, which is alphabetical), not with
        the order of ``stocks``.
    prices : pandas.DataFrame, optional
        Pre-fetched close prices. When provided no network call is made, which
        lets the caller run an out-of-sample backtest on a held-out window.
    start, end : datetime-like, optional
        Download window used only when ``prices`` is None.
    """
    if prices is None:
        if end is None:
            end = pd.Timestamp.now()
        if start is None:
            start = end - pd.DateOffset(years=1)
        prices = download_stock_data(stocks, start=start, end=end)

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    returns = prices.pct_change().dropna()

    daily_returns = (returns.values @ np.asarray(weights))
    portfolio_value = np.cumprod(1 + daily_returns)
    portfolio_value = np.concatenate([[1.0], portfolio_value])
    return portfolio_value


if __name__ == "__main__":
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    weights = np.array([0.4, 0.4, 0.2])
    print(backtest(stocks, weights))
