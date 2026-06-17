"""Tests for the Ledoit-Wolf covariance shrinkage estimator."""
import numpy as np
import pytest

from main import ledoit_wolf_cov


def _sample_cov_mle(X):
    Xc = X - X.mean(axis=0)
    t = X.shape[0]
    # errstate guards against spurious Accelerate-BLAS matmul FP flags (numpy 2.x).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (Xc.T @ Xc) / t


def test_shrinkage_intensity_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    _, delta = ledoit_wolf_cov(X)
    assert 0.0 <= delta <= 1.0


def test_result_is_symmetric_and_psd():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    sigma, _ = ledoit_wolf_cov(X)
    assert np.allclose(sigma, sigma.T)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() >= -1e-10


def test_diagonal_is_preserved():
    # The constant-correlation prior shares the sample diagonal, so a convex
    # combination must leave the variances unchanged.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    sigma, _ = ledoit_wolf_cov(X)
    assert np.allclose(np.diag(sigma), np.diag(_sample_cov_mle(X)))


def test_shrinkage_improves_conditioning_when_data_is_scarce():
    # With few observations relative to assets the sample covariance is
    # ill-conditioned; shrinkage should reduce the condition number.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 20))
    sigma, delta = ledoit_wolf_cov(X)
    sample = _sample_cov_mle(X)
    assert delta > 0
    assert np.linalg.cond(sigma) < np.linalg.cond(sample)


def test_shrinkage_shrinks_toward_zero_with_abundant_data():
    # More observations -> sample covariance is reliable -> less shrinkage.
    rng = np.random.default_rng(4)
    cov = np.array([[0.04, 0.01, 0.0],
                    [0.01, 0.05, 0.0],
                    [0.0, 0.0, 0.03]])
    L = np.linalg.cholesky(cov)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        small = (rng.normal(size=(50, 3)) @ L.T)
        large = (rng.normal(size=(5000, 3)) @ L.T)
    _, delta_small = ledoit_wolf_cov(small)
    _, delta_large = ledoit_wolf_cov(large)
    assert delta_large < delta_small


def test_requires_at_least_two_assets():
    with pytest.raises(ValueError):
        ledoit_wolf_cov(np.random.default_rng(5).normal(size=(50, 1)))
