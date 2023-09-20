import numpy as np
from package.core.slsqpb import slsqpb


def solve(
    m,
    meq,
    la,
    n,
    x,
    xl,
    xu,
    f,
    c,
    g,
    a,
    acc,
    iter,
    mode,
    w,
    l_w,
    sdat,
    ldat,
    alphamin,
    alphamax,
    tolf,
    toldf,
    toldx,
    max_iter_ls,
    nnls_mode,
    infinite_bound,
):
    infBnd = np.inf if infinite_bound == 0 else abs(infinite_bound)

    # Check length of working arrays
    n1 = n + 1
    mineq = m - meq + n1 + n1
    il = (
        (3 * n1 + m) * (n1 + 1)
        + (n1 - meq + 1) * (mineq + 2)
        + 2 * mineq
        + (n1 + mineq) * (n1 - meq)
        + 2 * meq
        + n1 * n // 2
        + 2 * m
        + 3 * n
        + 4 * n1
        + 1
    )
    im = max(mineq, n1 - meq)

    if l_w < il:
        mode = 1000 * max(10, il)
        mode += max(10, im)  # type: ignore
        iter = 0
        return

    if meq > n:
        mode = 2
        iter = 0
        return

    # Prepare data for calling sqpbdy - initial addresses in w
    im = 1
    il = im + max(1, m)
    il = im + la
    ix = il + n1 * n // 2 + 1
    ir = ix + n
    is_ = ir + n + n + max(1, m)
    is_ = ir + n + n + la
    iu = is_ + n1
    iv = iu + n1
    iw = iv + n1

    sdat["n1"] = n1

    slsqpb(
        m,
        meq,
        la,
        n,
        x,
        xl,
        xu,
        f,
        c,
        g,
        a,
        acc,
        iter,
        mode,
        w[ir - 1 :],
        w[il - 1 :],
        w[ix - 1 :],
        w[im - 1 :],
        w[is_ - 1 :],
        w[iu - 1 :],
        w[iv - 1 :],
        w[iw - 1 :],
        sdat["t"],
        sdat["f0"],
        sdat["h1"],
        sdat["h2"],
        sdat["h3"],
        sdat["h4"],
        sdat["n1"],
        sdat["n2"],
        sdat["n3"],
        sdat["t0"],
        sdat["gs"],
        sdat["tol"],
        sdat["line"],
        sdat["alpha"],
        sdat["iexact"],
        sdat["incons"],
        sdat["ireset"],
        sdat["itermx"],
        ldat,
        alphamin,
        alphamax,
        tolf,
        toldf,
        toldx,
        max_iter_ls,
        nnls_mode,
        infBnd,
    )
