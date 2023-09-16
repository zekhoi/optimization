import numpy as np
from . import helper, _compute
from optimization.helper import split_bounds

eps = np.finfo(float).eps


def calculate_gradient(func, x):
    epsilon = 1e-6  # Small value for numerical differentiation
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon

        gradient[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)

    return gradient


def slsqp_search_direction(
    objective_func, x0, constraints, bounds, epsilon=1e-6, max_iter=100
):
    """
    Compute the search direction using the SLSQP algorithm.

    return: x
    """

    bounds = helper.split_bounds(bounds)
    x = np.clip(x0, bounds[0], bounds[1])

    grad = _compute.compute_f_grad(objective_func, x0)
    g_grads = _compute.compute_g_grads(constraints, x0)
    jacobian = np.array(
        [calculate_gradient(constraint, x) for constraint in constraints]
    )

    print(g_grads)
    print(jacobian)
    cons = {"eq": (), "ineq": ()}
    for ic, constraint in enumerate(constraints):
        # create jac property if not exists
        type = constraint["type"]

        if "fun" not in constraint:
            raise ValueError("Constraint %d has no function defined." % ic)

        jac = constraint.get("jac")
        if jac is None:
            print("")

        # add jac property to constraints
        # cons[type] += (
        #     {"fun": constraint["fun"], "jac": jac, "args": constraint.get("args", ())},
        # )

    meq = sum(map(len, [np.atleast_1d(c["fun"](x, *c["args"])) for c in cons["eq"]]))
    mieq = sum(map(len, [np.atleast_1d(c["fun"](x, *c["args"])) for c in cons["ineq"]]))
    m = meq + mieq
    la = np.array([1, m]).max()
    n = len(x)

    # Calculate the Hessian matrix
    hess = np.zeros((n, n))


def objective_func(x):
    return x[0] ** 2 + x[1] ** 2


def equality_constraint(x):
    return np.array([x[0] + x[1] - 1])


def inequality_constraint(x):
    return np.array([x[0] - x[1]])


x = np.array([0.5, 0.5])

# List of constraints
constraints = [equality_constraint, inequality_constraint]

# Finding the direction
direction = slsqp_search_direction(
    objective_func, x, constraints, bounds=[(0, 1), (0, 1)]
)
print("Direction:", direction)
