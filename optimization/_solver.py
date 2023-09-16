import numpy as np


def new_x_th(x, alpha, d):
    return x + alpha * d


def find_direction(jac_constraints, grad_obj):
    A = np.dot(jac_constraints, jac_constraints.T)
    g = np.dot(jac_constraints, grad_obj)
    d = np.linalg.solve(A.T, -g.T)
    return d


def find_step_size(f, x, constraints, d, alpha=1.0, beta=0.5, max_iter=100):
    def phi(alpha):
        return f(x + alpha * d)

    def psi(alpha):
        return all(c["fun"](x + alpha * d) <= 0 for c in constraints)

    a = 0.0
    b = np.inf
    alpha_star = alpha

    for _ in range(max_iter):
        if not psi(alpha_star) or not phi(alpha_star) < phi(
            0
        ) + beta * alpha_star * np.dot(d, d):
            a = alpha_star
        else:
            break

        alpha_star = (a + b) / 2

    return alpha_star
