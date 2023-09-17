import numpy as np


def new_x_th(x, alpha, d):
    return x + alpha * d


def find_direction(jac_constraints, grad_obj):
    A = np.dot(jac_constraints, jac_constraints.T)
    g = np.dot(jac_constraints, grad_obj)
    d = np.linalg.solve(A.T, -g.T)
    return d


def find_step_size(
    f, x, constraints, d, bounds, alpha=0.1, rho=0.5, max_iter=100, c=0.1
):
    fx = f(x)

    dfx = np.dot(d, f(x))

    while np.any(f(x + alpha * d) > fx + c * alpha * dfx):
        alpha *= rho

    return alpha
