from numpy.linalg import inv
import numpy as np
from scipy.optimize import minimize   

# portfolio weight calculation functions
def tangency_weights(mu, sigma, rf = 0.0):
    excess_returns = mu - rf
    inv_sigma = inv(sigma)
    ones = np.ones(len(mu))

    weights_unnormalized = inv_sigma @ excess_returns
    weights = weights_unnormalized / (ones.T @ weights_unnormalized)

    return weights

def tangency_weights_constrained(mu, sigma, rf=0.0, bounds=None):
    n = len(mu)
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    

    def neg_sharpe(w):
        port_ret = w.dot(mu)
        port_vol = np.sqrt(w.dot(sigma).dot(w))

        if port_vol == 0:
            return 1e6
        return - (port_ret - rf) / port_vol
    

    cons = ({'type' : 'eq',
             'fun': lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-9, 'maxiter':1000})


    if not res.success:
        print("Optimization did not converge: ", res.message)

    return res.x



# Efficient frontier (min variance for target returns)
def efficient_frontier(mu, sigma, returns_range=None, points=50):
    n = len(mu)
    if returns_range is None:
        # min and max possible portfolio returns (using unconstrained min-variance and max-return)
        max_ret = mu.max()
        min_ret = mu.min()
        returns_range = np.linspace(min_ret, max_ret, points)
    frontier_weights = []
    frontier_rets = []
    frontier_vols = []

    def var_obj(w):
        return w.dot(sigma).dot(w)

    bounds = [(0.0,1.0)] * n  # change if you want allow shorting
    for target in returns_range:
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, target=target: w.dot(mu) - target}
        )
        x0 = np.ones(n) / n
        res = minimize(var_obj, x0=x0, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success:
            w = res.x
            frontier_weights.append(w)
            frontier_rets.append(w.dot(mu))
            frontier_vols.append(np.sqrt(w.dot(sigma).dot(w)))
        else:
            # fallback: skip this target
            continue

# fake change
    return np.array(frontier_rets), np.array(frontier_vols), np.array(frontier_weights)
