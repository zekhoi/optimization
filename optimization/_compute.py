import numpy as np


def objective_gradient(f, x, eps=1e-8):
    """
    Compute the gradient of the objective function using finite differences
    Ini kalo gak salah kalkulus 1 nyari turunan pertama dari fungsi
    """
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        gradient[i] = (f(x_plus) - f(x)) / 2 * eps

    return gradient


def constraint_gradient(f, x, eps=1e-8):
    """
    # Compute the gradient of the constrain function using finite differences
    Ini kalo gak salah kalkulus 1 nyari turunan pertama dari fungsi
    """
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        gradient[i] = (f(x_plus) - f(x)) / 2 * eps

    return gradient


def constraint_gradients(constraints, x, eps=1e-8):
    jacobian = np.zeros((len(constraints), len(x)))
    for i, constraint in enumerate(constraints):
        jacobian[i, :] = constraint_gradient(constraint["fun"], x)

    return jacobian


def lagrange(fx, jacobian, lambdas):
    return fx - np.dot(
        lambdas,
        jacobian,
    )
