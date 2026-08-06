"""Tests for the Black-Scholes option pricing and Greeks."""
import numpy as np
import pytest

from options import black_scholes_greeks


# A standard reference case used throughout: at-the-money, 1y, r=5%, vol=20%.
REF = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


def test_reference_call_and_put_prices():
    # Textbook values for S=K=100, T=1, r=5%, sigma=20%, q=0.
    call = black_scholes_greeks(**REF, option_type="call")
    put = black_scholes_greeks(**REF, option_type="put")
    assert call["price"] == pytest.approx(10.4506, abs=1e-3)
    assert put["price"] == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    # C - P == S - K*exp(-rT)  (q = 0).
    call = black_scholes_greeks(**REF, option_type="call")
    put = black_scholes_greeks(**REF, option_type="put")
    lhs = call["price"] - put["price"]
    rhs = REF["S"] - REF["K"] * np.exp(-REF["r"] * REF["T"])
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_greek_signs():
    call = black_scholes_greeks(**REF, option_type="call")
    put = black_scholes_greeks(**REF, option_type="put")

    # Call delta in (0, 1); put delta in (-1, 0).
    assert 0.0 < call["delta"] < 1.0
    assert -1.0 < put["delta"] < 0.0

    # Gamma and vega are positive for long options.
    assert call["gamma"] > 0 and put["gamma"] > 0
    assert call["vega"] > 0 and put["vega"] > 0

    # Long options lose value as time passes (theta < 0) for this case.
    assert call["theta"] < 0 and put["theta"] < 0

    # Call rho positive, put rho negative.
    assert call["rho"] > 0
    assert put["rho"] < 0


def test_gamma_and_vega_match_across_call_and_put():
    # Gamma and vega are identical for a call and put with the same terms.
    call = black_scholes_greeks(**REF, option_type="call")
    put = black_scholes_greeks(**REF, option_type="put")
    assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], rel=1e-12)


def test_call_put_delta_relationship():
    # With q = 0: delta_call - delta_put == 1.
    call = black_scholes_greeks(**REF, option_type="call")
    put = black_scholes_greeks(**REF, option_type="put")
    assert call["delta"] - put["delta"] == pytest.approx(1.0, abs=1e-9)


def test_delta_matches_finite_difference():
    # Analytic delta should match a central finite difference of the price.
    h = 1e-4
    base = black_scholes_greeks(**REF, option_type="call")["delta"]
    up = black_scholes_greeks(**{**REF, "S": REF["S"] + h}, option_type="call")["price"]
    dn = black_scholes_greeks(**{**REF, "S": REF["S"] - h}, option_type="call")["price"]
    fd = (up - dn) / (2 * h)
    assert base == pytest.approx(fd, abs=1e-5)


def test_vega_matches_finite_difference():
    # Vega is reported per 1% (1 percentage point) change in vol, so the finite
    # difference in price per 0.01 change in sigma should match.
    h = 1e-4
    analytic = black_scholes_greeks(**REF, option_type="call")["vega"]
    up = black_scholes_greeks(**{**REF, "sigma": REF["sigma"] + h}, option_type="call")["price"]
    dn = black_scholes_greeks(**{**REF, "sigma": REF["sigma"] - h}, option_type="call")["price"]
    fd = (up - dn) / (2 * h) * 0.01  # scale to per-1% to match the convention
    assert analytic == pytest.approx(fd, abs=1e-5)


def test_degenerate_inputs_return_none():
    # Expired or zero-vol contracts (junk yfinance IVs) must not blow up.
    for bad in [
        {**REF, "T": 0.0},
        {**REF, "T": -1.0},
        {**REF, "sigma": 0.0},
        {**REF, "S": 0.0},
        {**REF, "K": 0.0},
    ]:
        result = black_scholes_greeks(**bad, option_type="call")
        assert all(v is None for v in result.values())


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes_greeks(**REF, option_type="straddle")


def test_dividend_yield_reduces_call_value():
    # A positive dividend yield lowers a call's value relative to q = 0.
    no_div = black_scholes_greeks(**REF, option_type="call")["price"]
    with_div = black_scholes_greeks(**REF, option_type="call", q=0.03)["price"]
    assert with_div < no_div
