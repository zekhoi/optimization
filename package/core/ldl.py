import numpy as np


def solve(n, a, z, sigma, w):
    t = 0.0
    v = 0.0
    u = 0.0
    tp = 0.0
    beta = 0.0
    alpha = 0.0
    delta = 0.0
    gamma = 0.0

    if abs(sigma) > 0.0:
        ij = 0
        t = 1.0 / sigma
        if sigma <= 0.0:
            # Prepare negative update
            for i in range(n):
                w[i] = z[i]
            for i in range(n):
                v = w[i]
                t = t + v * v / a[ij]
                for j in range(i + 1, n):
                    ij = ij + 1
                    w[j] = w[j] - v * a[ij]
                ij = ij + 1
            if t >= 0.0:
                t = np.finfo(float).eps / sigma
            for i in range(n):
                j = n - i - 1
                ij = ij - i
                u = w[j]
                w[j] = t
                t = t - u * u / a[ij]
        # Here updating begins
        for i in range(n):
            v = z[i]
            delta = v / a[ij]
            if sigma < 0.0:
                tp = w[i]
            if sigma > 0.0:
                tp = t + delta * v
            alpha = tp / t
            a[ij] = alpha * a[ij]
            if i == n - 1:
                return
            beta = delta / tp
            if alpha > 4.0:
                gamma = t / tp
                for j in range(i + 1, n):
                    ij = ij + 1
                    u = a[ij]
                    a[ij] = gamma * u + beta * z[j]
                    z[j] = z[j] - v * u
            else:
                for j in range(i + 1, n):
                    ij = ij + 1
                    z[j] = z[j] - v * a[ij]
                    a[ij] = a[ij] + beta * z[j]
            ij = ij + 1
            t = tp
