import yfinance as yf
import pandas as pd
import numpy as np
from portfolio import tangency_weights, tangency_weights_constrained, efficient_frontier


# downloading and loading data
def download_stock_data(tickers, start, end):
    """Download adjusted close prices, always returning a DataFrame.

    yfinance returns a Series for a single ticker; we normalise to a one-column
    DataFrame so callers can rely on a consistent shape and on ``.columns``.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True)
    close = data['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close


def calculate_annualized_return(returns, periods_per_year=252):
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1


def calculate_mu_sigma(returns, annualized=True, periods_per_year=252):
    mu = returns.mean()
    sigma = returns.cov()
    if annualized:
        mu = mu * periods_per_year
        sigma = sigma * periods_per_year
    return mu.values, sigma.values


def ledoit_wolf_cov(returns):
    """Ledoit-Wolf shrinkage of the sample covariance matrix.

    Shrinks the (noisy) sample covariance toward a constant-correlation target,
    following Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance
    Matrix" (Journal of Portfolio Management). The optimal shrinkage intensity
    is estimated in closed form, so there are no tuning parameters and no extra
    dependencies.

    Parameters
    ----------
    returns : array-like, shape (t, n)
        Period (e.g. daily) asset returns, t observations of n assets.

    Returns
    -------
    sigma : numpy.ndarray, shape (n, n)
        The shrunk covariance matrix (same period scale as the input).
    shrinkage : float
        The estimated shrinkage intensity delta in [0, 1]; 0 means the sample
        covariance is used as-is, 1 means the constant-correlation target.
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (t, n)")
    t, n = X.shape
    if n < 2:
        raise ValueError("need at least 2 assets for shrinkage")

    # Demean and form the (MLE, 1/t) sample covariance. The matmuls below are
    # wrapped in errstate because Apple's Accelerate BLAS raises spurious
    # divide/overflow FP flags inside matmul on numpy 2.x; the math is exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        X = X - X.mean(axis=0)
        sample = (X.T @ X) / t

        var = np.diag(sample)
        sqrtvar = np.sqrt(var)
        outer_sqrt = np.outer(sqrtvar, sqrtvar)

        # Constant-correlation prior: average off-diagonal correlation.
        corr = sample / outer_sqrt
        rbar = (corr.sum() - n) / (n * (n - 1))
        prior = rbar * outer_sqrt
        np.fill_diagonal(prior, var)

        # pi-hat: sum of asymptotic variances of the sample covariance entries.
        Y = X ** 2
        phi_mat = (Y.T @ Y) / t - sample ** 2
        pi_hat = phi_mat.sum()

        # rho-hat: covariances between sample variances and sample covariances,
        # weighted by the constant-correlation structure.
        term1 = ((X ** 3).T @ X) / t
        theta_mat = term1 - var[:, None] * sample
        np.fill_diagonal(theta_mat, 0.0)
        weight = np.outer(1.0 / sqrtvar, sqrtvar)  # (i, j) -> sqrt(var_j / var_i)
        rho_hat = np.diag(phi_mat).sum() + rbar * (weight * theta_mat).sum()

        # gamma-hat: misfit between sample covariance and the prior.
        gamma_hat = np.sum((sample - prior) ** 2)

    if gamma_hat <= 0:
        return sample, 0.0

    kappa = (pi_hat - rho_hat) / gamma_hat
    shrinkage = float(max(0.0, min(1.0, kappa / t)))

    sigma = shrinkage * prior + (1.0 - shrinkage) * sample
    return sigma, shrinkage


if __name__ == "__main__":
    stocks = download_stock_data(("AAPL", "MSFT"), '2023-01-01', '2024-01-01')
    print(stocks.head())
