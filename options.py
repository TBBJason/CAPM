"""Option chain retrieval and Black-Scholes Greeks.

yfinance exposes option chains (strike, bid/ask, implied volatility, volume,
open interest) but does *not* provide the Greeks. This module fetches the chain
and computes price sensitivities ourselves with the Black-Scholes-Merton model,
using the implied volatility that yfinance reports for each contract.

Greek conventions (trader-facing scaling):
  * delta  -- per $1 move in the underlying
  * gamma  -- change in delta per $1 move in the underlying
  * vega   -- per 1 percentage-point (1%) change in implied volatility
  * theta  -- per calendar day of time decay (negative for long options)
  * rho    -- per 1 percentage-point (1%) change in the risk-free rate
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# Fields we surface from the raw yfinance chain (everything else is dropped).
_CHAIN_FIELDS = [
    "contractSymbol",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
    "inTheMoney",
]

# One calendar year in days; used to scale theta to a per-day figure.
_DAYS_PER_YEAR = 365.0


def black_scholes_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes-Merton price and Greeks for a single European option.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded risk-free rate (e.g. 0.04 for 4%).
    sigma : float
        Implied volatility (annualised, e.g. 0.25 for 25%).
    option_type : {"call", "put"}
        Contract type.
    q : float, optional
        Continuous dividend yield of the underlying (default 0).

    Returns
    -------
    dict
        ``price`` plus ``delta``, ``gamma``, ``vega``, ``theta`` and ``rho``.
        Returns ``None`` for every Greek when the inputs are degenerate
        (non-positive time, vol, spot or strike), which happens for expired or
        illiquid contracts where yfinance reports a junk implied vol.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    # Guard against degenerate inputs: expired contracts (T<=0) or the junk
    # implied vols (~1e-5) yfinance returns for illiquid strikes make the
    # formulas blow up or divide by zero, so we bail out with Nones instead.
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return {
            "price": None, "delta": None, "gamma": None,
            "vega": None, "theta": None, "rho": None,
        }

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    disc_r = np.exp(-r * T)   # risk-free discount factor
    disc_q = np.exp(-q * T)   # dividend discount factor
    pdf_d1 = norm.pdf(d1)

    # Gamma and vega are identical for calls and puts.
    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:  # put
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        # Scale to the conventional trader-facing units (see module docstring).
        "vega": float(vega / 100.0),
        "theta": float(theta / _DAYS_PER_YEAR),
        "rho": float(rho / 100.0),
    }


def _underlying_price(ticker):
    """Best-effort spot price for ``ticker``.

    fast_info is quick but occasionally returns None, so we fall back to the
    last daily close before giving up.
    """
    tk = ticker if isinstance(ticker, yf.Ticker) else yf.Ticker(ticker)
    for getter in (
        lambda: tk.fast_info.get("last_price"),
        lambda: tk.fast_info.get("lastPrice"),
        lambda: tk.history(period="1d")["Close"].iloc[-1],
    ):
        try:
            price = getter()
            if price:
                return float(price)
        except Exception:
            continue
    return None


def fetch_expirations(ticker):
    """Return available option expiration dates and the underlying spot price.

    Returns
    -------
    dict
        ``{"ticker", "underlying_price", "expirations": [...]}``.
    """
    tk = yf.Ticker(ticker)
    expirations = list(tk.options or [])
    return {
        "ticker": ticker,
        "underlying_price": _underlying_price(tk),
        "expirations": expirations,
    }


def _years_to_expiry(expiry):
    """Fraction of a year from now until ``expiry`` (an ISO date string).

    Options expire at the market close, so we add one day to the expiry date;
    this keeps T strictly positive on expiration day itself.
    """
    now = pd.Timestamp.now(tz=None).normalize()
    exp = pd.Timestamp(expiry).normalize() + pd.Timedelta(days=1)
    return max((exp - now).total_seconds() / (365.0 * 24 * 3600), 0.0)


def _enrich_side(df, S, T, r, option_type, q=0.0):
    """Trim a raw yfinance chain side and attach computed Greeks per contract."""
    rows = []
    for _, row in df.iterrows():
        rec = {f: (None if pd.isna(row.get(f)) else row.get(f)) for f in _CHAIN_FIELDS}
        # Cast the numeric/bool fields to plain Python types for clean JSON.
        for key in ("strike", "lastPrice", "bid", "ask", "impliedVolatility"):
            rec[key] = None if rec[key] is None else float(rec[key])
        for key in ("volume", "openInterest"):
            rec[key] = None if rec[key] is None else int(rec[key])
        rec["inTheMoney"] = bool(rec["inTheMoney"]) if rec["inTheMoney"] is not None else None

        sigma = rec["impliedVolatility"]
        greeks = black_scholes_greeks(
            S, rec["strike"], T, r, sigma if sigma else 0.0, option_type, q=q
        )
        rec["bs_price"] = greeks.pop("price")
        rec.update(greeks)
        rows.append(rec)
    return rows


def fetch_option_chain(ticker, expiry, rf=0.04, q=0.0):
    """Fetch one expiry's option chain with Black-Scholes Greeks attached.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    expiry : str
        Expiration date (ISO ``YYYY-MM-DD``); must be one of the dates returned
        by :func:`fetch_expirations`.
    rf : float, optional
        Annual risk-free rate used in the Greeks (default 0.04).
    q : float, optional
        Continuous dividend yield (default 0).

    Returns
    -------
    dict
        Underlying price, time to expiry, and ``calls``/``puts`` lists where
        each contract carries its Greeks plus a model ``bs_price``.
    """
    tk = yf.Ticker(ticker)
    S = _underlying_price(tk)
    if S is None:
        raise ValueError(f"Could not determine underlying price for {ticker}")

    chain = tk.option_chain(expiry)  # raises if the expiry is invalid
    T = _years_to_expiry(expiry)

    return {
        "ticker": ticker,
        "expiry": expiry,
        "underlying_price": S,
        "risk_free_rate": rf,
        "years_to_expiry": T,
        "calls": _enrich_side(chain.calls, S, T, rf, "call", q=q),
        "puts": _enrich_side(chain.puts, S, T, rf, "put", q=q),
    }
