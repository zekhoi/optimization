from scipy.optimize import minimize
import numpy as np

prices = [1.47, 1.23, 1.12, 0.97, 8.7]
flash_point = [0.5, 0.5, 0.5, 0.5, 0.5]
pour_point = [0.5, 0.5, 0.5, 0.5, 0.5]
FP_min = 0.5
PP_max = 0.5


def f(x):
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
    x = np.array(xk)
    print("Iteration x:", xk)


x0_no_additive = [0.2, 0.2, 0.2, 0.2, 0.2]
x0_with_additive = [0.24, 0.24, 0.24, 0.24, 0.04]


# Define constraints
constraints = (
    {"type": "ineq", "fun": g6},
    {"type": "ineq", "fun": g7},
    {"type": "eq", "fun": g8},
)

bounds = [(0, 1)] * 4 + [(0.04, 1)]

# Optimization using SciPy's SLSQP
result = minimize(
    f,
    x0_with_additive,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    # callback print x at each iteration
    callback=callback_function,
    options={"disp": True},
)

# Extract results
optimal_x = result.x
optimal_objective_value = result.fun

# print("Optimal solution:")
# print("x =", optimal_x)
# print("Objective value =", optimal_objective_value)
