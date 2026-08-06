# 📊 Portfolio Optimizer

A web app that builds the **maximum-Sharpe (tangency) portfolio** for a set of
stocks using Modern Portfolio Theory, plots the **efficient frontier**, and runs
an **out-of-sample backtest**. Expected returns and covariances are estimated
from historical prices, with optional **Ledoit–Wolf covariance shrinkage** for
more stable, realistic weights.

**Live demo:** https://capm-sga6.onrender.com/

---

## Features

- **Tangency portfolio optimization** — maximizes the Sharpe ratio, long-only or
  with short selling allowed.
- **Efficient frontier** — the min-variance frontier plotted with the tangency
  point and the capital market line.
- **Out-of-sample backtest** — optimizes on a training window and backtests on a
  held-out final year, so reported performance isn't in-sample.
- **Ledoit–Wolf shrinkage** — optional constant-correlation covariance shrinkage
  to reduce estimation error (recommended; on by default).
- **Interactive UI** — add tickers, set the risk-free rate, lookback period, and
  toggles; view metrics, allocations, the frontier, and the backtest chart.
- **Options chain & Greeks** — pull a live option chain for any ticker and view
  the full **Black–Scholes Greeks** (delta, gamma, vega, theta, rho) computed
  per contract, since `yfinance` provides implied vol but not the Greeks.

---

## Tech stack

- **Backend:** Flask (`backend.py`), served by gunicorn in production.
- **Math:** NumPy / SciPy (`portfolio.py`, `main.py`).
- **Data:** `yfinance` for historical prices.
- **Frontend:** single-page vanilla JS + Chart.js (`index.html`).

---

## Project layout

| File | Purpose |
|------|---------|
| `backend.py` | Flask app, API endpoints, request handling, orchestration. |
| `portfolio.py` | Optimization math: tangency weights (closed-form & constrained), efficient frontier. |
| `main.py` | Data download (`yfinance`) and estimators, including Ledoit–Wolf shrinkage. |
| `options.py` | Option chain retrieval and Black–Scholes pricing/Greeks. |
| `backtest.py` | Cumulative portfolio-value backtest (accepts pre-fetched prices). |
| `index.html` | Frontend UI and charts. |
| `test_portfolio.py`, `test_api.py`, `test_estimators.py`, `test_options.py` | Pytest suite. |
| `requirements.txt` / `requirements-dev.txt` | Runtime / dev dependencies. |
| `Procfile` | Production start command for Render/Heroku-style hosts. |

---

## Running locally

```bash
git clone https://github.com/TBBJason/CAPM.git
cd CAPM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend.py
```

Then open http://localhost:10000.

The port defaults to `10000` and can be overridden with the `PORT` environment
variable.

---

## API

All endpoints accept/return JSON. Bodies are `POST`ed; `GET` returns a status
message.

### `POST /api/optimize`

Request:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "rf": 0.04,
  "lookback_years": 3,
  "allow_shorting": false,
  "use_shrinkage": true
}
```

Response (abridged):

```json
{
  "weights": { "AAPL": 0.41, "GOOG": 0.22, "MSFT": 0.37 },
  "expected_return": 0.18,
  "volatility": 0.21,
  "sharpe": 0.67,
  "shrinkage_intensity": 0.34,
  "backtest_values": [1.0, 1.002, ...],
  "frontier": { "rf": 0.04, "frontier": [...], "tangency": {...}, "assets": [...] }
}
```

### `POST /api/frontier`

Same request shape; returns only the efficient frontier, tangency point, and
per-asset risk/return points.

### `POST /api/options/expirations`

Request: `{ "ticker": "AAPL" }`

Returns the underlying spot price and the list of available expiration dates:

```json
{
  "ticker": "AAPL",
  "underlying_price": 231.4,
  "expirations": ["2026-08-07", "2026-08-14", "..."]
}
```

### `POST /api/options/chain`

Request:

```json
{ "ticker": "AAPL", "expiry": "2026-08-14", "rf": 0.04, "dividend_yield": 0.0 }
```

Returns the calls and puts for that expiry, each contract enriched with a
Black–Scholes theoretical price (`bs_price`) and Greeks:

```json
{
  "ticker": "AAPL",
  "expiry": "2026-08-14",
  "underlying_price": 231.4,
  "years_to_expiry": 0.023,
  "calls": [
    {
      "strike": 230.0, "lastPrice": 3.1, "bid": 3.0, "ask": 3.2,
      "impliedVolatility": 0.28, "inTheMoney": true, "bs_price": 3.05,
      "delta": 0.54, "gamma": 0.06, "vega": 0.12, "theta": -0.08, "rho": 0.01
    }
  ],
  "puts": [ ... ]
}
```

**Greek conventions:** delta and gamma are per **$1** move in the underlying,
vega is per **1 percentage-point** change in implied volatility, theta is per
**calendar day** of decay, and rho is per **1 percentage-point** change in the
risk-free rate. Greeks are `null` for contracts with a degenerate/missing
implied volatility (common on very illiquid strikes).

### `GET /api/health`

Liveness check.

**Note on weight labels:** `yfinance` returns price columns in alphabetical
order, so all weights are labeled by the actual data columns (not the input
order) to avoid mislabeling.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the optimization math (hand-checked tangency weights), the Ledoit–Wolf
estimator invariants, and the API endpoints (with mocked price data, no
network).

---

## Deployment (Render)

Render builds from the GitHub repo: it installs `requirements.txt` and starts
the app via the `Procfile` (`gunicorn backend:app`). Pushing to the deployed
branch triggers a redeploy.

---

## Caveats

This is an educational tool, not investment advice. Mean-variance optimization
is sensitive to estimation error in historical returns; shrinkage and the
out-of-sample backtest mitigate but do not eliminate this. Past performance does
not predict future results.
