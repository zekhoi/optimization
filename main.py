import numpy as np


def f(x):
    # or sum(x[i] * prices[i] for i in range(len(x)))
    # or np.sum(x * prices)
    return np.dot(x, prices)


prices = np.array([1.2, 0.98])
x = [0.7, 0.3]

fx = f(x)

print(fx)
