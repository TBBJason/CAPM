import yfinance as yf
import pandas as pd
import numpy as np



# downloading and loading data
def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']


#Helper functions for calculations
def calculate_returns(prices):
    return ((prices / prices.shift(1)) - 1).dropna()

def calculate_annualized_return(returns, periods_per_year=252):
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

def calculate_mu_sigma(returns, annualized=True, periods_per_year=252):
    mu = returns.mean()
    sigma = returns.std()
    if annualized:
        mu = mu * periods_per_year
        sigma = sigma * np.sqrt(periods_per_year)
    return mu, sigma




if __name__ == "__main__":
    # data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
    # print(data.head())
    stocks = download_stock_data(("AAPL", "MSFT"), '2023-01-01', '2024-01-01')

    # print(aapl.head())

    returns = calculate_returns(stocks)
    print("expected returns: ", returns.head())

    expected_returns = calculate_annualized_return(returns)
    print("expected returns: ", expected_returns)
    mu, sigma = calculate_mu_sigma(returns)
    print("mu", mu, "sigma", sigma)