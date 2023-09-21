import numpy as np
from scipy.optimize._differentiable_functions import ScalarFunction
from scipy.optimize._numdiff import approx_derivative
from optimization import helper

FD_METHODS = ("2-point", "3-point", "cs")


def rhs_matrix(x, gradient, constraints_eq, constraints_ineq):
    num_variables = len(x)
    num_eq_constraints = len(constraints_eq)
    num_ineq_constraints = len(constraints_ineq)
    rhs = np.zeros(num_variables + num_eq_constraints + num_ineq_constraints)
    rhs[:num_variables] = -gradient
    rhs[num_variables : num_variables + num_eq_constraints] = np.array(
        [-constraint["fun"](x) for constraint in constraints_eq]
    )
    rhs[num_variables + num_eq_constraints :] = np.array(
        [-constraint["fun"](x) for constraint in constraints_ineq]
    )

    return rhs


def kkt_matrix(constraint_gradients, lagrange_multipliers):
    """
    Construct the KKT matrix.
    """
    n_variables = constraint_gradients.shape[1]
    n_constraints = constraint_gradients.shape[0]

    kkt_matrix = np.zeros((n_variables + n_constraints, n_variables + n_constraints))
    kkt_matrix[:n_variables, :n_variables] = np.eye(n_variables)
    kkt_matrix[:n_variables, n_variables:] = constraint_gradients.T
    kkt_matrix[n_variables:, :n_variables] = constraint_gradients
    kkt_matrix[n_variables:, n_variables:] = np.diag(lagrange_multipliers)

    return kkt_matrix


def objective_gradient(f, x):
    """
    Compute the objective function value and its gradient at x.
    """
    epsilon = 1e-6  # Small value for numerical differentiation
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        gradient[i] = (f(x_plus) - f(x)) / epsilon

    return gradient


def jacobian(constrains, x):
    n = len(x)
    m = len(constrains)
    jacobian = np.zeros((m, n))
    eps = 1
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = 1e-6
        jacobian[:, i] = (
            np.array([constrain["fun"](x + perturbation) for constrain in constrains])
            - np.array([constrain["fun"](x - perturbation) for constrain in constrains])
        ) / (  # type: ignore
            2 * eps
        )  # type: ignore
    return jacobian


def hessian(f_grad, x):
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(len(x)):
        for j in range(len(x)):
            hessian[i, j] = approx_derivative(
                f_grad, x, method="2-point", rel_step=1e-6, f0=None
            )[i][j]
    return hessian


def constraint_gradients(constraints, x):
    epsilon = 1e-6  # Small value for numerical differentiation
    gradients = []

    for constraint in constraints:
        gradient = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            gradient[i] = (constraint["fun"](x_plus) - constraint["fun"](x)) / epsilon
        gradients.append(gradient)

    return gradients


def scalar_function(
    fun,
    x0,
    jac=None,
    args=(),
    bounds=None,
    epsilon=None,
    finite_diff_rel_step=None,
    hess=None,  # type: ignore
):
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        grad = "2-point"
        epsilon = epsilon

    if hess is None:

        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    sf = ScalarFunction(
        fun, x0, args, grad, hess, finite_diff_rel_step, bounds, epsilon=epsilon
    )

    return sf


def new_constraints(constraints, bounds):
    cons = {"eq": (), "ineq": ()}
    for ic, constraint in enumerate(constraints):
        # create jac property if not exists
        type = constraint["type"]

        if "fun" not in constraint:
            raise ValueError("Constraint %d has no function defined." % ic)

        jac = lambda x, *args: new_jac(constraint["fun"], x, bounds)

        cons[type] += (  # type: ignore
            {"fun": constraint["fun"], "jac": jac, "args": constraint.get("args", ())},
        )
    return cons


def new_jac(fun, x, bounds):
    x = helper.clip_x(x, bounds)
    return approx_derivative(
        fun, x, method="3-point", args=(), rel_step=None, bounds=bounds
    )
