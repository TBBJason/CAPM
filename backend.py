from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.linalg import inv
from portfolio import tangency_weights_constrained, tangency_weights
from main import download_stock_data
from backtest import backtest

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

@app.route('/api/optimize', methods=['POST', 'GET'])
def optimize():
    """
    POST: Receive optimization request from frontend
    GET: Health check
    """
    try:
        if request.method == 'GET':
            return jsonify({'status': 'ok', 'message': 'POST data to this endpoint to optimize'})

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        tickers = data.get('tickers', [])
        rf = data.get('rf', 0.04)
        lookback_years = data.get('lookback_years', 3)

        print(f"\n{'='*60}")
        print(f"Received optimization request")
        print(f"  Tickers: {tickers}")
        print(f"  Risk-free rate: {rf*100:.1f}%")
        print(f"  Lookback period: {lookback_years} years")
        print(f"{'='*60}")

        if len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400

        # Fetch historical data
        end_date = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        start_date = end_date - pd.DateOffset(years=lookback_years)
        
        print(f"Downloading historical data ({start_date.date()} to {end_date.date()})...")
        prices = download_stock_data(tickers, start=start_date, end=end_date)
        
        if prices.isnull().all().all():
            return jsonify({'error': 'No data available for tickers'}), 400
        
        returns = ((prices / prices.shift(1)) - 1).dropna()
        print(f"  Downloaded {len(returns)} trading days")

        # Calculate mu, sigma
        mu = returns.mean() * 252
        sigma = returns.cov() * 252

        print(f"Calculating optimal weights...")
        # Optimize
        weights = tangency_weights(mu.values, sigma.values, rf=rf)
        weights = np.maximum(weights, 0)  # long-only
        weights /= weights.sum()

        # Display weights
        for ticker, w in zip(tickers, weights):
            print(f"  {ticker}: {w*100:6.2f}%")

        # Calculate metrics
        port_ret = float(weights @ mu.values)
        port_vol = float(np.sqrt(weights @ sigma.values @ weights))
        sharpe = float((port_ret - rf) / port_vol) if port_vol > 0 else 0

        print(f"\nPortfolio Metrics:")
        print(f"  Annual Return: {port_ret*100:.2f}%")
        print(f"  Annual Volatility: {port_vol*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.4f}")

        # Backtest (1 year)
        portfolio_value = backtest(tickers, weights)

        final_value = portfolio_value[-1]
        total_return = (final_value - 1) * 100


        return jsonify({
            'weights': {ticker: float(w) for ticker, w in zip(tickers, weights)},
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe': sharpe,
            'backtest_values': portfolio_value.tolist()
        })

    except Exception as e:
        print(f" RROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

if __name__ == '__main__':
    print("Listening on: http://localhost:5000")
    print("API endpoint: POST http://localhost:5000/api/optimize")
    # print("Health check: GET http://localhost:5000/api/health")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, threaded=True)
