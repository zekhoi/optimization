import numpy as np
from package.core import h12, ldp


def solve(e, f, g, h, le, me, lg, mg, n, x, xnorm, w, mode, max_iter_ls, nnls_mode):
    epmach = np.finfo(float).eps
    one = 1.0

    for i in range(1, n + 1):
        j = min(i + 1, n)
        t = np.dot(e[:, i - 1], e[:, j - 1])
        h12.solve(1, i, i + 1, me, e[:, i - 1], 1, t, e[:, j - 1], 1, le, n - i + 1)
        h12.solve(2, i, i + 1, me, e[:, i - 1], 1, t, f, 1, 1, 1)

    mode = 5
    for i in range(1, mg + 1):
        for j in range(1, n + 1):
            if abs(e[j - 1, j - 1]) < epmach or np.isnan(e[j - 1, j - 1]):
                return
            g[i - 1, j - 1] = (
                g[i - 1, j - 1] - np.dot(g[i - 1, : j - 1], e[: j - 1, j - 1])
            ) / e[j - 1, j - 1]
        h[i - 1] = h[i - 1] - np.dot(g[i - 1, :], f)

    ldp.solve(g, lg, mg, n, h, x, xnorm, w, mode, max_iter_ls, nnls_mode)
    if mode == 1:
        np.add(x, f, out=x)
        for i in range(n, 0, -1):
            j = min(i + 1, n)
            x[i - 1] = (x[i - 1] - np.dot(e[i - 1, j - 1 :], x[j - 1 :])) / e[
                i - 1, i - 1
            ]
        j = min(n + 1, me)
        t = norm(f[j - 1 :])  # type: ignore
        xnorm = np.sqrt(xnorm**2 + t**2)
