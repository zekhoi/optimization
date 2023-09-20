import numpy as np


def solve(mode, lpivot, l1, m, iue, u, up, c, ice, icv, ncv):
    zero = 0.0
    one = 1.0

    if not (0 < lpivot < l1 or l1 > m):
        return

    cl = abs(u[0, lpivot - 1])
    if mode != 2:
        # Construct the transformation
        for j in range(l1, m + 1):
            cl = max(abs(u[0, j - 1]), cl)
        if cl <= zero:  # type: ignore
            return
        clinv = one / cl  # type: ignore
        sm = (u[0, lpivot - 1] * clinv) ** 2
        for j in range(l1, m + 1):
            sm += (u[0, j - 1] * clinv) ** 2
        cl = cl * np.sqrt(sm)
        if u[0, lpivot - 1] > zero:
            cl = -cl
        up[0] = u[0, lpivot - 1] - cl
        u[0, lpivot - 1] = cl
    elif cl <= zero:
        return

    if ncv > 0:
        # Apply the transformation i + u * (u**t) / b to c
        b = up[0] * u[0, lpivot - 1]
        # b must be nonpositive here
        if b < zero:
            b = one / b
            i2 = 0 - icv + ice * (lpivot - 1)
            incr = ice * (l1 - lpivot)
            for j in range(ncv):
                i2 += icv
                i3 = i2 + incr
                i4 = i3
                sm = c[i2] * up[0]
                for i in range(l1, m + 1):
                    sm += c[i3] * u[0, i - 1]
                    i3 += ice
                if abs(sm) > zero:
                    sm = sm * b
                    c[i2] += sm * up[0]
                    for i in range(l1, m + 1):
                        c[i4] += sm * u[0, i - 1]
                        i4 += ice
