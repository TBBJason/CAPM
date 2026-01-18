from main import download_stock_data

def backtest(stocks):
    start_date = pd.timestamp.now()
    end_date = start_date - pd.dateOffset(year=1)
    download_stock_data(stocks, start=start)