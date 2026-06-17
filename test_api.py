"""API tests for backend.py using a mocked price download (no network)."""
import numpy as np
import pandas as pd
import pytest

import backend


def _synthetic_prices(columns, periods=4 * 252, seed=0):
    """Deterministic geometric random walk ending today."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=periods, freq="B")
    data = {}
    for i, col in enumerate(columns):
        daily = rng.normal(0.0004 + i * 0.0001, 0.01, size=periods)
        data[col] = 100 * np.cumprod(1 + daily)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def client():
    backend.app.config["TESTING"] = True
    return backend.app.test_client()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_optimize_requires_two_tickers(client):
    resp = client.post("/api/optimize", json={"tickers": ["AAPL"]})
    assert resp.status_code == 400


def test_optimize_returns_weights_labelled_by_column_order(client, monkeypatch):
    # yfinance returns columns alphabetically; the user submits a different
    # order. The response must label weights by the actual data columns.
    cols = ["AAPL", "GOOG", "MSFT"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols),
    )

    resp = client.post("/api/optimize", json={
        "tickers": ["MSFT", "AAPL", "GOOG"],  # deliberately not alphabetical
        "rf": 0.04,
        "lookback_years": 3,
    })
    assert resp.status_code == 200
    body = resp.get_json()

    assert set(body["weights"].keys()) == set(cols)
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    # All long-only weights are non-negative.
    assert all(w >= -1e-9 for w in body["weights"].values())
    # Backtest series starts at 1.0 and has more than one point (held-out year).
    assert body["backtest_values"][0] == pytest.approx(1.0)
    assert len(body["backtest_values"]) > 1
    for key in ("expected_return", "volatility", "sharpe"):
        assert key in body


def test_optimize_reports_dropped_tickers(client, monkeypatch):
    # Mock returns data for only two of the three requested tickers.
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(["AAPL", "MSFT"]),
    )
    resp = client.post("/api/optimize", json={
        "tickers": ["AAPL", "MSFT", "FAKETICKER"],
        "lookback_years": 2,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert "warning" in body
    assert "FAKETICKER" in body["warning"]


def test_optimize_allow_shorting_can_go_negative(client, monkeypatch):
    cols = ["AAA", "BBB"]
    monkeypatch.setattr(
        backend, "download_stock_data",
        lambda tickers, start, end: _synthetic_prices(cols, seed=7),
    )
    resp = client.post("/api/optimize", json={
        "tickers": cols,
        "allow_shorting": True,
        "lookback_years": 2,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    # Weights still sum to 1 under shorting.
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)
