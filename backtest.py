from main import download_stock_data
import pandas as pd
import numpy as np

def backtest(stocks, weights):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    prices = download_stock_data(stocks, start=start_date, end=end_date)
    returns = ((prices / prices.shift(1)) - 1).dropna()

    daily_returns = (returns @ weights).values
    portfolio_value = np.cumprod(1 + daily_returns)
    # print(portfolio_value)

    # print("=" * 40)
    portfolio_value = np.concatenate([[1.0], portfolio_value])  
    # print(portfolio_value)
    return portfolio_value

if __name__ == "__main__":
    # Example usage
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    weights = np.array([0.4, 0.4, 0.2])
    backtest(stocks, weights)