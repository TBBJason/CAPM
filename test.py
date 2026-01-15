import numpy as np

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

v = np.array([4, 5, 6])

mu = np.array([0.1, 0.2, 0.15])
risk_free_rate = 0.03

sigma = np.array([[0.005, -0.010, 0.004],
                  [-0.010, 0.040, -0.002],
                  [0.004, -0.002, 0.023]])
excess_returns = mu - risk_free_rate
inv_sigma = np.linalg.inv(sigma)


result = A @ v
print(result)