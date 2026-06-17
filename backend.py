from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from portfolio import (
    tangency_weights,
    tangency_weights_constrained,
    efficient_frontier,
)
from main import download_stock_data
from backtest import backtest
import os
from flask import send_from_directory

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Server-side guard rails (the client also caps these, but never trust the client).
MAX_TICKERS = 10
TRADING_DAYS = 252
BACKTEST_YEARS = 1        # length of the held-out, out-of-sample backtest window
FRONTIER_POINTS = 60      # number of target returns sampled along the frontier


class RequestError(Exception):
    """Raised for client-side (4xx) problems with a request."""


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
    prices = prices.dropna(axis=1, how="all")          # drop invalid symbols
    prices = prices.ffill().dropna(axis=0, how="any")  # fill the odd missing day
    return prices


def _parse_request(data):
    """Pull and validate shared parameters from a request body."""
    if not data:
        raise RequestError("No JSON data provided")
    tickers = data.get("tickers", [])
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    if len(tickers) < 2:
        raise RequestError("Need at least 2 tickers")
    if len(tickers) > MAX_TICKERS:
        raise RequestError(f"Maximum {MAX_TICKERS} tickers allowed")
    return {
        "tickers": tickers,
        "rf": data.get("rf", 0.04),
        "lookback_years": data.get("lookback_years", 3),
        "allow_shorting": bool(data.get("allow_shorting", False)),
    }


def _load_market_data(tickers, lookback_years):
    """Download prices and return (columns, mu, sigma, dropped, test_prices).

    Trains on the lookback window ending one year ago and holds out the most
    recent year (``test_prices``) for out-of-sample use. All returned arrays
    follow ``columns`` order, which is the (alphabetical) yfinance column order.
    """
    test_end = pd.Timestamp.now()
    split = test_end - pd.DateOffset(years=BACKTEST_YEARS)
    train_start = split - pd.DateOffset(years=lookback_years)

    prices = download_stock_data(tickers, start=train_start, end=test_end)
    prices = _clean_prices(prices)
    if prices.empty or prices.shape[1] < 2:
        raise RequestError("Not enough valid tickers with price data")

    dropped = [t for t in tickers if t not in prices.columns]
    columns = list(prices.columns)

    train_prices = prices.loc[:split]
    test_prices = prices.loc[split:]

    train_returns = train_prices.pct_change().dropna()
    if len(train_returns) < 2:
        raise RequestError("Not enough history in the lookback window")

    mu = train_returns.mean() * TRADING_DAYS
    sigma = train_returns.cov() * TRADING_DAYS
    return columns, mu, sigma, dropped, test_prices


def _solve_weights(mu, sigma, rf, allow_shorting):
    if allow_shorting:
        return tangency_weights(mu.values, sigma.values, rf=rf)
    return tangency_weights_constrained(mu.values, sigma.values, rf=rf)


def _compute_frontier(columns, mu, sigma, rf, allow_shorting):
    """Build the efficient frontier, tangency point, and per-asset points."""
    rets, vols, _ = efficient_frontier(
        mu.values, sigma.values, points=FRONTIER_POINTS, allow_shorting=allow_shorting
    )
    frontier = [
        {"volatility": float(v), "return": float(r)}
        for v, r in zip(vols, rets)
    ]

    w_t = _solve_weights(mu, sigma, rf, allow_shorting)
    t_ret = float(w_t @ mu.values)
    t_vol = float(np.sqrt(w_t @ sigma.values @ w_t))
    t_sharpe = float((t_ret - rf) / t_vol) if t_vol > 0 else 0.0

    asset_vols = np.sqrt(np.diag(sigma.values))
    assets = [
        {"ticker": col, "volatility": float(av), "return": float(m)}
        for col, av, m in zip(columns, asset_vols, mu.values)
    ]

    return {
        "rf": float(rf),
        "frontier": frontier,
        "tangency": {"volatility": t_vol, "return": t_ret, "sharpe": t_sharpe},
        "assets": assets,
    }


@app.route('/api/optimize', methods=['POST', 'GET'])
def optimize():
    try:
        if request.method == 'GET':
            return jsonify({'status': 'ok', 'message': 'POST data to this endpoint to optimize'})

        params = _parse_request(request.json)
        tickers = params["tickers"]
        rf = params["rf"]
        allow_shorting = params["allow_shorting"]

        columns, mu, sigma, dropped, test_prices = _load_market_data(
            tickers, params["lookback_years"]
        )

        weights = _solve_weights(mu, sigma, rf, allow_shorting)

        port_ret = float(weights @ mu.values)
        port_vol = float(np.sqrt(weights @ sigma.values @ weights))
        sharpe = float((port_ret - rf) / port_vol) if port_vol > 0 else 0.0

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
            # Frontier is included so the UI can plot it without a second call.
            'frontier': _compute_frontier(columns, mu, sigma, rf, allow_shorting),
        }
        if dropped:
            response['warning'] = f"No data for: {', '.join(dropped)}"
        return jsonify(response)

    except RequestError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/frontier', methods=['POST', 'GET'])
def frontier():
    """Return only the efficient frontier, tangency point, and asset points."""
    try:
        if request.method == 'GET':
            return jsonify({'status': 'ok', 'message': 'POST tickers to this endpoint for the efficient frontier'})

        params = _parse_request(request.json)
        columns, mu, sigma, dropped, _ = _load_market_data(
            params["tickers"], params["lookback_years"]
        )
        response = _compute_frontier(
            columns, mu, sigma, params["rf"], params["allow_shorting"]
        )
        if dropped:
            response['warning'] = f"No data for: {', '.join(dropped)}"
        return jsonify(response)

    except RequestError as e:
        return jsonify({'error': str(e)}), 400
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
