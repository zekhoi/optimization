from scipy.optimize import minimize
import numpy as np
import random

from utils.generator import generate_x, generate_bounds

# Define constants and initial values
base_oils_counts = 4

treat_rates = [
    0.01,
    0.1,
]
x0 = generate_x(base_oils_counts)

x0_with_additive = generate_x(base_oils_counts, treat_rates)
# prices = [random.uniform(0.5, 1.5) for _ in range(base_oils_counts)] + [
#     random.uniform(5, 10) for _ in range(len(treat_rates))
# ]
prices = [0.49, 0.51, 1.31, 0.98, 5.2, 4.7]
flash_point = [
    random.uniform(0.5, 1.5) for _ in range(base_oils_counts + len(treat_rates))
]
pour_point = [
    random.uniform(0.5, 1.5) for _ in range(base_oils_counts + len(treat_rates))
]
FP_min = 0.5
PP_max = 0.5

# Define objective function, constraints functions and bounds


def f(x):
    # or sum(x[i] * prices[i] for i in range(len(x)))
    # or np.sum(x * prices)
    return np.dot(x, prices)


def g6(x):
    return (
        sum(
            x[i] * 51708 * np.exp((np.log(flash_point[i]) - 2.6287) ** 2 / (-0.91725))
            for i in range(len(x))
        )
        - FP_min
    )


def g7(x):
    return sum(x[i] * 3262000 * (pour_point[i] / 1000) for i in range(len(x))) - PP_max


def g8(x):
    return sum(x) - 1


def callback_function(xk):
    print("Iteration x:", ["{:.10f}".format(x) for x in xk])
    print("Total:", sum(xk))


constraints = (
    {"type": "ineq", "fun": g6},
    {"type": "ineq", "fun": g7},
    {"type": "eq", "fun": g8},
)

bounds = generate_bounds(base_oils_counts, treat_rates)
tolerance = 1e-6

# Optimization using SciPy's SLSQP

print("Initial solution:", x0_with_additive)
print("Prices:", prices)
print("Tolerances:", tolerance)
print("Initial objective value:", f(x0_with_additive))
print(
    "Initial constraints values:",
    g6(x0_with_additive),
    g7(x0_with_additive),
    g8(x0_with_additive),
)
print("Bounds:", bounds)
print()
print("Iteration x:", x0_with_additive)
result = minimize(
    f,
    x0_with_additive,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    tol=tolerance,
    callback=callback_function,
    options={"disp": True},
)

# Extract results
optimal_x = result.x
optimal_objective_value = result.fun

# print("Optimal solution:")
# print("x =", optimal_x)
# print("Objective value =", optimal_objective_value)
