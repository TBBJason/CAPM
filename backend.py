from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from portfolio import tangency_weights_constrained, tangency_weights
from main import download_stock_data
from backtest import backtest
import os
from flask import send_from_directory

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Server-side guard rails (the client also caps these, but never trust the client).
MAX_TICKERS = 10
TRADING_DAYS = 252
BACKTEST_YEARS = 1  # length of the held-out, out-of-sample backtest window


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response


def _clean_prices(prices):
    """Normalise yfinance output to a DataFrame and drop tickers with no data."""
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    # Drop any ticker whose entire column is NaN (e.g. invalid symbol).
    prices = prices.dropna(axis=1, how="all")
    # Forward/back fill the occasional missing day so returns stay aligned.
    prices = prices.ffill().dropna(axis=0, how="any")
    return prices


@app.route('/api/optimize', methods=['POST', 'GET'])
def optimize():
    try:
        if request.method == 'GET':
            return jsonify({'status': 'ok', 'message': 'POST data to this endpoint to optimize'})

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        tickers = data.get('tickers', [])
        rf = data.get('rf', 0.04)
        lookback_years = data.get('lookback_years', 3)
        allow_shorting = data.get('allow_shorting', False)

        # Normalise + validate ticker input.
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
        if len(tickers) < 2:
            return jsonify({'error': 'Need at least 2 tickers'}), 400
        if len(tickers) > MAX_TICKERS:
            return jsonify({'error': f'Maximum {MAX_TICKERS} tickers allowed'}), 400

        # Train on the lookback window ending one year ago; hold out the most
        # recent year for an honest, out-of-sample backtest.
        test_end = pd.Timestamp.now()
        split = test_end - pd.DateOffset(years=BACKTEST_YEARS)
        train_start = split - pd.DateOffset(years=lookback_years)

        prices = download_stock_data(tickers, start=train_start, end=test_end)
        prices = _clean_prices(prices)

        if prices.empty or prices.shape[1] < 2:
            return jsonify({'error': 'Not enough valid tickers with price data'}), 400

        # Report any tickers that yfinance could not resolve.
        dropped = [t for t in tickers if t not in prices.columns]

        # IMPORTANT: yfinance returns columns alphabetically, so all downstream
        # arrays (mu, sigma, weights) follow prices.columns order, NOT the user
        # input order. Always label results with prices.columns.
        train_prices = prices.loc[:split]
        test_prices = prices.loc[split:]

        train_returns = train_prices.pct_change().dropna()
        if len(train_returns) < 2:
            return jsonify({'error': 'Not enough history in the lookback window'}), 400

        mu = train_returns.mean() * TRADING_DAYS
        sigma = train_returns.cov() * TRADING_DAYS

        # Optimize (max-Sharpe tangency portfolio).
        weights_unconstrained = tangency_weights(mu.values, sigma.values, rf=rf)
        if allow_shorting:
            weights = weights_unconstrained
        else:
            weights = tangency_weights_constrained(mu.values, sigma.values, rf=rf)

        columns = list(prices.columns)

        # Portfolio metrics (model expectations from the training window).
        port_ret = float(weights @ mu.values)
        port_vol = float(np.sqrt(weights @ sigma.values @ weights))
        sharpe = float((port_ret - rf) / port_vol) if port_vol > 0 else 0.0

        # Out-of-sample backtest on the held-out window.
        if len(test_prices) > 1:
            portfolio_value = backtest(columns, weights, prices=test_prices)
        else:
            portfolio_value = np.array([1.0])

        response = {
            'weights': {col: float(w) for col, w in zip(columns, weights)},
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe': sharpe,
            'backtest_values': portfolio_value.tolist(),
        }
        if dropped:
            response['warning'] = f"No data for: {', '.join(dropped)}"
        return jsonify(response)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
