from numpy.linalg import inv
import numpy as np

# portfolio weight calculation functions
def tangency_weights(mu, sigma, rf = 0.0):
    excess_returns = mu - rf
    inv_sigma = inv(sigma)
    ones = np.ones(len(mu))

    weights_unnormalized = inv_sigma @ excess_returns
    weights = weights_unnormalized / (ones.T @ weights_unnormalized)

    return weights

