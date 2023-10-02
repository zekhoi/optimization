import numpy as np


# Define the Lagrange function L(x, lambda) as a function of x, lambda, f(x), and g(x)
def lagrange_function(x, lambda_, f, g):
    return f(x) - np.sum(lambda_ * g(x))


# Define the objective function f(x)
def objective_function(x):
    return x[0] ** 2 + x[1] ** 2


# Define the constraint functions g(x)
def constraint_functions(x):
    return np.array([x[0] + x[1] - 1, -x[0], -x[1]])


# Number of variables (n) and constraints (m)
n = 2  # Number of variables
m_rho_alt = 2  # Number of constraints for which rho.alt applies
m = 3  # Total number of constraints (m_rho.alt + m-m_rho.alt)

# Initialize Lagrange multipliers lambda and the current solution x^k
lambda_ = np.zeros(m)
x_k = np.array([1.0, 1.0])  # Initial solution


# Compute the gradient of the objective function nabla f(x)
def gradient_objective_function(x):
    return np.array([2 * x[0], 2 * x[1]])


# Compute the gradient of the constraint functions nabla g(x)
def gradient_constraint_functions(x):
    return np.array([[1, 1], [-1, 0], [0, -1]])


# Compute the Hessian matrix B = nabla_xx^2 L(x, lambda)
def hessian_B(x, lambda_, gradient_objective, gradient_constraints):
    B = np.zeros((n, n))
    for i in range(m_rho_alt):
        grad_g_i_x = gradient_constraints[i]
        B += lambda_[i] * np.outer(grad_g_i_x, grad_g_i_x)
    return B


# Compute the gradient of the Lagrangian function nabla L(x, lambda)
def gradient_L(x, lambda_, gradient_objective, gradient_constraints):
    grad_L = gradient_objective(x)
    for i in range(m):
        grad_L -= lambda_[i] * gradient_constraints[i]
    return grad_L


# Compute the Hessian matrix B = nabla_xx^2 L(x, lambda)
B = hessian_B(
    x_k, lambda_, gradient_objective_function, gradient_constraint_functions(x_k)
)

# Compute the gradient of the Lagrangian function nabla L(x, lambda)
grad_L_xk = gradient_L(
    x_k, lambda_, gradient_objective_function, gradient_constraint_functions(x_k)
)

# Solve the quadratic programming subproblem to find the search direction d
d = np.linalg.solve(B, -grad_L_xk)

# Print the search direction
print("Search Direction (d):", d)
