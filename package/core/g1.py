import numpy as np


def g1(a, b):
    sig = np.zeros(1)
    c = np.zeros(1)
    s = np.zeros(1)

    xr = 0.0
    yr = 0.0

    if abs(a) > abs(b):
        xr = b / a
        yr = np.sqrt(1.0 + xr**2)
        c[0] = np.sign(1.0 / yr, a)
        s[0] = c[0] * xr
        sig[0] = abs(a) * yr
    else:
        if abs(b) > 0.0:
            xr = a / b
            yr = np.sqrt(1.0 + xr**2)
            s[0] = np.sign(1.0 / yr, b)
            c[0] = s[0] * xr
            sig[0] = abs(b) * yr
        else:
            sig[0] = 0.0
            c[0] = 0.0
            s[0] = 1.0

    return c[0], s[0], sig[0]
