import yfinance as yf
import pandas as pd
import numpy as np



# downloading and loading data
def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']


#Helper functions for calculations
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_annualized_return(log_returns, periods_per_year=252):
    return (1 + log_returns).prod() ** (periods_per_year / len(log_returns)) - 1

def calculate_mu_sigma(log_returns, annualized=True, periods_per_year=252):
    mu = log_returns.mean()
    sigma = log_returns.std()
    if annualized:
        mu = mu * periods_per_year
        sigma = sigma * np.sqrt(periods_per_year)
    return mu, sigma





if __name__ == "__main__":
    # data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
    # print(data.head())
    stocks = download_stock_data(("AAPL", "MSFT"), '2020-01-01', '2021-01-01')

    # print(aapl.head())

    log_returns = calculate_log_returns(stocks)
    print("expected log returns: ", log_returns.head())

    expected_log_returns = calculate_annualized_return(log_returns)
    print("expected log returns: ", expected_log_returns)

    mu, sigma = calculate_mu_sigma(log_returns)
    print("mu", mu, "sigma", sigma)