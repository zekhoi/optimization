import numpy as np


def approx_grad(func, x):
    eps = 1e-6  # Small perturbation for numerical differentiation
    grad = np.gradient(func(x), eps)
    return grad


def approx_hessian(func, x):
    eps = 1e-6  # Small perturbation for numerical differentiation
    grad_x = np.gradient(func(x), eps)
    grad_y = np.gradient(grad_x, eps)
    hessian = np.diag(grad_y)
    return hessian


def get_qp_bounds(x, bounds):
    qp_bounds = []
    for bound in bounds:
        lower_bound, upper_bound = bound
        qp_bound = (max(lower_bound, x), min(upper_bound, x))
        qp_bounds.append(qp_bound)
    return qp_bounds


def get_qp_constraints(x, constraints):
    qp_constraints = []
    for constraint, bound in constraints.items():
        lower_bound, upper_bound = bound
        constraint_value = constraint(x)
        qp_constraint = (
            max(lower_bound - constraint_value, 0),
            max(constraint_value - upper_bound, 0),
        )
        qp_constraints.append(qp_constraint)
    return qp_constraints


def line_search(func, x, p):
    alpha = 1.0  # Initial step length
    beta = 0.5  # Step length reduction factor
    c = 0.1  # Sufficient decrease parameter

    f = func(x)
    g = approx_grad(func, x)
    while func(x + alpha * p) > f + c * alpha * np.dot(g, p):
        alpha *= beta

    ls_x = x + alpha * p
    ls_f = func(ls_x)
    return ls_x, ls_f


def converged(x, x_old, tol=1e-6):
    return np.max(np.abs(x - x_old)) < tol
