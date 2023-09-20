import numpy as np


def solve(
    mode, ax, bx, f, tol, a, b, d, e, p, q, r, u, v, w, x, m, fu, fv, fw, fx, tol1, tol2
):
    c = (3.0 - np.sqrt(5.0)) / 2.0  # golden section ratio = 0.381966011
    sqrteps = np.sqrt(np.finfo(float).eps)  # square root of machine precision

    if mode == 1:
        fx = f
        fv = fx
        fw = fv
    elif mode == 2:
        fu = f
        if fu > fx:
            if u < x:
                a = u
            if u >= x:
                b = u
            if fu <= fw or np.abs(w - x) <= 0.0:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or np.abs(v - x) <= 0.0 or np.abs(v - w) <= 0.0:
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            if u < x:
                b = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
    else:
        a = ax
        b = bx
        e = 0.0
        v = a + c * (b - a)
        w = v
        x = w
        linmin = x
        mode = 1
        return linmin

    m = 0.5 * (a + b)
    tol1 = sqrteps * np.abs(x) + tol
    tol2 = tol1 + tol1

    if np.abs(x - m) <= tol2 - 0.5 * (b - a):
        linmin = x
        mode = 3
    else:
        r = 0.0
        q = r
        p = q
        if np.abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = q - r
            q = q + q
            if q > 0.0:
                p = -p
            if q < 0.0:
                q = -q
            r = e
            e = d

        if np.abs(p) >= 0.5 * np.abs(q * r) or p <= q * (a - x) or p >= q * (b - x):
            if x >= m:
                e = a - x
            if x < m:
                e = b - x
            d = c * e
        else:
            d = p / q
            if u - a < tol2:
                d = np.sign(tol1, m - x)
            if b - u < tol2:
                d = np.sign(tol1, m - x)

        if np.abs(d) < tol1:
            d = np.sign(tol1, d)  # type: ignore
        u = x + d
        linmin = u
        mode = 2

    return linmin
