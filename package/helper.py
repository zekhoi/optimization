import numpy as np


def destroy_slsqpb_data(me):
    me = None


def destroy_linmin_data(me):
    me = None


def enforce_bounds(x, xl, xu, infbnd):
    if len(x) != len(xl) or len(x) != len(xu):
        raise ValueError("Dimensions of `x`, `xl`, and `xu` must be the same.")

    for i in range(len(x)):
        if x[i] < xl[i] and xl[i] > -infbnd and not np.isnan(xl[i]):
            x[i] = xl[i]
        elif x[i] > xu[i] and xu[i] < infbnd and not np.isnan(xu[i]):
            x[i] = xu[i]


def check_convergence(
    n,
    f,
    f0,
    x,
    x0,
    s,
    h3,
    acc,
    tolf,
    toldf,
    toldx,
    converged,
    not_converged,
    inconsistent_linearization,
):
    if h3 >= acc or inconsistent_linearization or np.isnan(f):
        mode = not_converged
    else:
        ok = False
        if not ok:
            ok = abs(f - f0) < acc
        if not ok:
            ok = np.linalg.norm(s) < acc
        if not ok and tolf >= 0:
            ok = abs(f) < tolf
        if not ok and toldf >= 0:
            ok = abs(f - f0) < toldf
        if not ok and toldx >= 0:
            xmx0 = x - x0
            ok = np.linalg.norm(xmx0) < toldx

        if ok:
            mode = converged
        else:
            mode = not_converged

    return mode


def daxpy(n, da, dx, incx, dy, incy):
    if n <= 0 or abs(da) == 0:
        return

    if incx == 1 and incy == 1:
        # Code for both increments equal to 1
        m = n % 4
        if m != 0:
            for i in range(1, m + 1):
                dy[i - 1] = dy[i - 1] + da * dx[i - 1]
            if n < 4:
                return
        mp1 = m + 1
        for i in range(mp1, n + 1, 4):
            dy[i - 1] = dy[i - 1] + da * dx[i - 1]
            dy[i] = dy[i] + da * dx[i]
            dy[i + 1] = dy[i + 1] + da * dx[i + 1]
            dy[i + 2] = dy[i + 2] + da * dx[i + 2]
            dy[i + 3] = dy[i + 3] + da * dx[i + 3]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx < 0:
            ix = (-n + 1) * incx + 1
        if incy < 0:
            iy = (-n + 1) * incy + 1
        for i in range(1, n + 1):
            dy[iy - 1] = dy[iy - 1] + da * dx[ix - 1]
            ix = ix + incx
            iy = iy + incy


def dcopy(n, dx, incx, dy, incy):
    if n <= 0:
        return

    if incx == 1 and incy == 1:
        # Code for both increments equal to 1
        m = n % 7
        if m != 0:
            for i in range(1, m + 1):
                dy[i - 1] = dx[i - 1]
            if n < 7:
                return
        mp1 = m + 1
        for i in range(mp1, n + 1, 7):
            dy[i - 1] = dx[i - 1]
            dy[i] = dx[i]
            dy[i + 1] = dx[i + 1]
            dy[i + 2] = dx[i + 2]
            dy[i + 3] = dx[i + 3]
            dy[i + 4] = dx[i + 4]
            dy[i + 5] = dx[i + 5]
            dy[i + 6] = dx[i + 6]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx < 0:
            ix = (-n + 1) * incx + 1
        if incy < 0:
            iy = (-n + 1) * incy + 1
        for i in range(1, n + 1):
            dy[iy - 1] = dx[ix - 1]
            ix = ix + incx
            iy = iy + incy


def ddot(n, dx, incx, dy, incy):
    ddot_result = 0.0
    dtemp = 0.0

    if n <= 0:
        return ddot_result

    if incx == 1 and incy == 1:
        # Code for both increments equal to 1
        m = n % 5
        if m != 0:
            for i in range(1, m + 1):
                dtemp += dx[i - 1] * dy[i - 1]
            if n < 5:
                return dtemp
        mp1 = m + 1
        for i in range(mp1, n + 1, 5):
            dtemp += (
                dx[i - 1] * dy[i - 1]
                + dx[i] * dy[i]
                + dx[i + 1] * dy[i + 1]
                + dx[i + 2] * dy[i + 2]
                + dx[i + 3] * dy[i + 3]
            )
        ddot_result = dtemp

    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx < 0:
            ix = (-n + 1) * incx + 1
        if incy < 0:
            iy = (-n + 1) * incy + 1
        for i in range(1, n + 1):
            dtemp += dx[ix - 1] * dy[iy - 1]
            ix = ix + incx
            iy = iy + incy
        ddot_result = dtemp

    return ddot_result
