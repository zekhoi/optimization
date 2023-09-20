import numpy as np
from package.core import lsei


def lsq(
    m, meq, n, nl, la, x, y, l, g, a, b, xl, xu, mode, max_iter_ls, nnls_mode, infbnd
):
    n1 = n + 1
    mineq = m - meq
    m1 = mineq + n + n

    # Determine whether to solve the problem with inconsistent linearization (n2=1) or not (n2=0)
    n2 = n1 * n // 2 + 1
    if n2 == nl:
        n2 = 0
    else:
        n2 = 1
    n3 = n - n2

    # Initialize arrays
    w = np.zeros(m1 + n3 + n3, dtype=np.float64)
    xnorm = 0.0

    i2 = 0
    i3 = 0
    i4 = 0
    ie = 0
    if_ = n * n + 1

    for i in range(n3):
        i1 = n1 - i
        diag = np.sqrt(l[i2])
        w[i3] = 0.0
        w[i3 : i3 + i1] = 0.0
        w[i3 : i3 + i1 - n2] = diag * l[i2]
        w[i3 : i3 + i1 - n2] *= diag
        w[i3] = diag
        w[if_ - 1 + i] = (g[i] - np.dot(w[i4 : i4 + i - 1], w[if_])) / diag

        i2 += i1 - n2
        i3 += n1
        i4 += n

    if n2 == 1:
        w[i3] = l[nl - 1]
        w[i4] = 0.0
        w[i3 : i3 + n3] = 0.0
        w[if_ - 1 + n] = 0.0

    w[if_ : if_ + n] *= -1.0

    ic = if_ + n
    id_ = ic + meq * n

    if meq > 0:
        for i in range(meq):
            w[ic - 1 + i] = a[i, :la] @ w[ic - 1 : ic - 1 + i + la]
        w[id_ : id_ + meq] *= -1.0

    ig = id_ + meq

    if mineq > 0:
        for i in range(mineq):
            w[ig - 1 + i] = a[meq + i, :la] @ w[ig - 1 : ig - 1 + i + m1]

    ih = ig + m1 * n
    iw = ih + mineq + 2 * n

    if mineq > 0:
        w[ih : ih + mineq] = b[meq : meq + mineq]
        w[ih : ih + mineq] *= -1.0

    # Augment matrix g by +i and -i, and augment vector h by xl and xu
    # NaN or infBnd value indicates no bound
    ip = ig + mineq
    il = ih + mineq
    num_unbounded = 0

    for i in range(n):
        if np.isnan(xl[i]) or xl[i] <= -infbnd:
            num_unbounded += 1
        else:
            w[il - 1] = 1.0 * xl[i]
            w[ip - 1 : ip - 1 + n] = 0.0
            w[ip - 1 + m1 * np.arange(n)] = 0.0
            w[ip - 1 + m1 * (i - 1)] = 1.0
            ip += 1
            il += 1

    for i in range(n):
        if np.isnan(xu[i]) or xu[i] >= infbnd:
            num_unbounded += 1
        else:
            w[il - 1] = -1.0 * xl[i]
            w[ip - 1 : ip - 1 + n] = 0.0
            w[ip - 1 + m1 * np.arange(n)] = 0.0
            w[ip - 1 + m1 * (i - 1)] = -1.0
            ip += 1
            il += 1

    # Call lsei function here or import it from package.core
    lsei.solve(
        w[ic - 1],
        w[id_],
        w[ie - 1],
        w[if_ - 1],
        w[ig - 1],
        w[ih - 1],
        max(1, meq),
        meq,
        n,
        n,
        m1,
        m1 - num_unbounded,
        n,
        x,
        xnorm,
        w[iw],
        mode,
        max_iter_ls,
        nnls_mode,
    )

    if mode == 1:
        # Restore lagrange multipliers (only for user-defined variables)
        y[:m] = w[iw - 1 : iw - 1 + m]
        if n3 > 0:
            # Set the rest of the multipliers to NaN (they are not used)
            y[m] = np.nan
            y[m + 1 : m + n3 + n3] = np.nan
        # Call enforce_bounds function here to ensure that bounds are not violated


# Call the lsq function with appropriate arguments here
