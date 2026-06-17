"""Tests for the portfolio optimisation math in portfolio.py."""
import numpy as np
import pytest

from portfolio import (
    tangency_weights,
    tangency_weights_constrained,
    efficient_frontier,
)


def test_tangency_weights_diagonal_hand_checked():
    # Uncorrelated assets, rf=0. For a diagonal covariance the tangency
    # weights are proportional to mu_i / var_i, then normalised to sum to 1.
    #   w1/w2 = (0.10/0.04) / (0.20/0.09) = 2.5 / 2.2222... = 1.125
    #   => w2 = 1 / 2.125 = 0.470588..., w1 = 0.529412...
    mu = np.array([0.10, 0.20])
    sigma = np.array([[0.04, 0.0],
                      [0.0, 0.09]])
    w = tangency_weights(mu, sigma, rf=0.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.5294117647, rel=1e-6)
    assert w[1] == pytest.approx(0.4705882353, rel=1e-6)


def test_tangency_weights_sum_to_one_three_assets():
    mu = np.array([0.12, 0.10, 0.07])
    sigma = np.array([[0.10, 0.01, 0.02],
                      [0.01, 0.08, 0.015],
                      [0.02, 0.015, 0.05]])
    w = tangency_weights(mu, sigma, rf=0.03)
    assert w.sum() == pytest.approx(1.0)


def test_constrained_is_long_only_and_normalised():
    # Asset 3 has high return but the unconstrained solution can short others;
    # the constrained optimiser must stay within [0, 1] and sum to 1.
    mu = np.array([0.05, 0.06, 0.30])
    sigma = np.array([[0.04, 0.03, 0.00],
                      [0.03, 0.05, 0.00],
                      [0.00, 0.00, 0.09]])
    w = tangency_weights_constrained(mu, sigma, rf=0.02)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-9)
    assert np.all(w <= 1.0 + 1e-9)


def test_constrained_respects_custom_bounds():
    mu = np.array([0.20, 0.05])
    sigma = np.array([[0.05, 0.0],
                      [0.0, 0.05]])
    bounds = [(0.0, 0.4), (0.0, 1.0)]
    w = tangency_weights_constrained(mu, sigma, rf=0.0, bounds=bounds)
    assert w[0] <= 0.4 + 1e-6
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_efficient_frontier_shapes_and_sanity():
    mu = np.array([0.12, 0.10, 0.07])
    sigma = np.array([[0.10, 0.01, 0.02],
                      [0.01, 0.08, 0.015],
                      [0.02, 0.015, 0.05]])
    rets, vols, weights = efficient_frontier(mu, sigma, points=20)
    assert len(rets) == len(vols) == len(weights)
    assert np.all(vols >= 0)
    # Returns should fall within the asset return range.
    assert rets.min() >= mu.min() - 1e-6
    assert rets.max() <= mu.max() + 1e-6
    # Each frontier portfolio's weights sum to 1.
    for w in weights:
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
