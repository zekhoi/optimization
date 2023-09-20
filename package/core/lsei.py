import numpy as np
from package.core import h12, lsi, hfti


def solve(
    c,
    d,
    e,
    f,
    g,
    h,
    lc,
    mc,
    le,
    me,
    lg,
    mg,
    n,
    x,
    xnrm,
    w,
    mode,
    max_iter_ls,
    nnls_mode,
):
    epmach = np.finfo(float).eps
    zero = 0.0

    mode = 2
    if mc <= n:
        l = n - mc
        mc1 = mc + 1
        iw = (l + 1) * (mg + 2) + 2 * mg + mc
        ie = iw + mc + 1
        if_ = ie + me * l
        ig = if_ + me

        dum = np.zeros(1, dtype=np.float64)  # Define dum
        krank = 0  # Define krank

        for i in range(1, mc + 1):
            j = min(i + 1, lc)
            h12.solve(
                1,
                i,
                i + 1,
                n,
                c[i - 1, :lc],
                lc,
                w[iw + i - 1],
                c[j - 1, :lc],
                lc,
                1,
                mc - i + 1,
            )
            h12.solve(2, i, i + 1, n, c[i - 1, :lc], lc, w[iw + i - 1], e, le, 1, me)
            h12.solve(2, i, i + 1, n, c[i - 1, :lc], lc, w[iw + i - 1], g, lg, 1, mg)

        mode = 6
        for i in range(1, mc + 1):
            if abs(c[i - 1, i - 1]) < epmach:
                return
            x[i - 1] = (d[i - 1] - np.dot(c[i - 1, :lc], x[:lc])) / c[i - 1, i - 1]

        mode = 1
        w[mc1 - 1] = zero

        if mc != n:
            for i in range(1, me + 1):
                w[if_ - 1 + i - 1] = f[i - 1] - np.dot(e[i - 1, :mc], x[:mc])

            for i in range(1, me + 1):
                w[ie - 1 + i - 1 : ie - 1 + i - 1 + me] = e[
                    i - 1, mc1 - 1 : mc1 - 1 + me
                ]

            for i in range(1, mg + 1):
                w[ig - 1 + i - 1 : ig - 1 + i - 1 + mg] = g[
                    i - 1, mc1 - 1 : mc1 - 1 + mg
                ]

            if mg > 0:
                for i in range(1, mg + 1):
                    h[i - 1] = h[i - 1] - np.dot(g[i - 1, :n], x[:n])
                lsi.solve(
                    w[ie - 1 :],
                    w[if_ - 1 :],
                    w[ig - 1 :],
                    h,
                    me,
                    me,
                    mg,
                    mg,
                    l,
                    x[mc1 - 1 :],
                    xnrm,
                    w[mc1 - 1 :],
                    mode,
                    max_iter_ls,
                    nnls_mode,
                )
                if mc == 0:
                    return
                t = np.linalg.norm(x[:mc])
                xnrm = np.sqrt(xnrm * xnrm + t * t)
                if mode != 1:
                    return
            else:
                mode = 7
                k = max(le, n)
                t = np.sqrt(epmach)
                hfti.solve(
                    w[ie - 1 :], me, me, l, w[if_ - 1 :], k, 1, t, krank, dum, w, w[l:]
                )
                xnrm = dum[0]
                w[mc1 - 1 :] = w[if_ - 1 :]
                if krank != l:
                    return
                mode = 1

        for i in range(1, me + 1):
            f[i - 1] = np.dot(e[i - 1, :n], x[:n]) - f[i - 1]

        for i in range(mc, 0, -1):
            h12.solve(2, i, i + 1, n, c[i - 1, :lc], lc, w[iw + i - 1], x, 1, 1, 1)

        for i in range(mc, 0, -1):
            j = min(i + 1, lc)
            w[i - 1] = (
                d[i - 1] - np.dot(c[i - 1, j - 1 :], w[j - 1 : j - 1 + mc])
            ) / c[i - 1, i - 1]
