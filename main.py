import optimization
import numpy as np
import cvxpy as cp
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import minimize
from optimization.table import d2270
from scipy.interpolate import interp1d


# Define constants and initial values
base_oils_counts = 5

treat_rates = [
    # 0.01,
    # 0.421,
]

x0 = optimization.helper.generate_x(base_oils_counts, treat_rates)

prices = optimization.helper.generate_prices(base_oils_counts, treat_rates)
v40 = optimization.helper.generate_viscosity(base_oils_counts, treat_rates)
v100 = optimization.helper.generate_viscosity(base_oils_counts, treat_rates)
flash_point = optimization.helper.generate_flash_points(base_oils_counts, treat_rates)
pour_point = optimization.helper.generate_pour_points(base_oils_counts, treat_rates)
v40_min = 121
v40_max = 180
v100_min = 100
v100_max = 200
FP_min = 100
PP_max = 10

prices = [1.2, 0.98]
x0 = [0.7, 0.3]
base_oils_counts = len(x0)
flash_point = [100, 200]
pour_point = [-100, -80]
VI = 80
v40 = [50, 70]
v100 = [80, 80]

# Define objective function, constraints functions and bounds


def f(x):
    return np.dot(x, prices)


def kv(target):
    def calculate_viscosity(x, target, v100, v40):
        def transform_viscosity(v):
            Z = v + 0.7
            W = np.log10(np.log10(Z))
            return W

        def transform_temperature(t_C):
            T = np.log10(t_C + 273.15)
            return T

        def inverse_slope(T0, T1, W0, W1):
            m_inv = (T1 - T0) / (W1 - W0)
            return m_inv

        def transform_specs(x, y):
            W0 = transform_viscosity(x)
            T0 = transform_temperature(100)
            W1 = transform_viscosity(y)
            T1 = transform_temperature(40)
            m_inv = inverse_slope(T0, T1, W0, W1)
            return W0, T0, m_inv

        def untransform_viscosity(WB):
            vB = np.power(10, np.power(10, WB)) - 0.7
            return vB

        WB_numerator = []
        WB_denominator = []
        for i in range(len(x)):
            W0, T0, m_inv = transform_specs(v100[i], v40[i])
            f = x[i]
            numerator_summand = f * (m_inv * W0 - T0)
            denominator_summand = f * m_inv
            WB_numerator = WB_numerator + [numerator_summand]
            WB_denominator = WB_denominator + [denominator_summand]

        TB = transform_temperature(target)
        WB = (TB + sum(WB_numerator)) / sum(WB_denominator)
        vB = untransform_viscosity(WB)
        return vB

    return lambda x: calculate_viscosity(x, target, v100, v40)


def g3_base(vB_40, vB_100) -> float:
    def get_L_and_H_values_from_table(vB_100):
        kv100_array = d2270["kin_viscosity_100c"]
        L_array = d2270["L"]
        H_array = d2270["H"]

        L_interp = interp1d(kv100_array, L_array)
        H_interp = interp1d(kv100_array, H_array)
        return L_interp(vB_100), H_interp(vB_100)

    U = vB_40
    Y = vB_100
    if 2 <= Y <= 70:
        L, H = get_L_and_H_values_from_table(vB_100)
    else:
        L = 0.8353 * np.power(Y, 2) + 14.67 * Y - 216
        H = 0.1684 * np.power(Y, 2) + 11.85 * Y - 97

    if U > H:
        return ((L - U) / (L - H)) * 100
    elif U < H:
        N = (np.log10(H) - np.log10(U)) / np.log10(Y)
        return (np.power(10, N) - 1) / 0.00715 + 100
    elif U == H:
        return 100
    else:
        return 100


def g1_min(x):
    return kv(40)(x) - v40_min
    # return kv(40)(x)


def g1_max(x):
    return v40_max - kv(40)(x)
    # return kv(40)(x)


def g2_min(x):
    return kv(100)(x) - v100_min
    # return kv(100)(x)


def g2_max(x):
    return v100_max - kv(100)(x)
    # return kv(100)(x)


def g3(x):
    return g3_base(kv(40)(x), kv(100)(x)) - VI


def g4(x):
    fp_in_fahrenheit_blended = sum(
        x[i] * 51708 * np.exp(((np.log(flash_point[i]) - 2.6287) ** 2) / (-0.91725))
        for i in range(len(x))
    )
    fp_in_fahrenheit = np.exp(
        np.power((-0.91725 * np.log(fp_in_fahrenheit_blended / 51708)), 0.5) + 2.6287
    )
    fp_in_celcius = optimization.helper.fahrenheit_to_celcius(fp_in_fahrenheit)
    return fp_in_celcius - FP_min  # - FP_min
    # return fp_in_celcius


def g5(x):
    pp_in_rankine_blended = sum(
        x[i]
        * 3262000
        * np.power(
            (optimization.helper.celcius_to_rankine(pour_point[i]) / 1000), 1 / 0.08
        )
        for i in range(len(x))
    )
    pp_in_rankine = np.power(pp_in_rankine_blended / 3262000, 0.08) * 1000
    pp_in_celcius = optimization.helper.rankine_to_celcius(pp_in_rankine)
    return PP_max - pp_in_celcius  # PP_max -
    # return pp_in_celcius


def g6(x):
    return sum(x) - 1
    # return sum(x)


def g(x):
    return np.array(
        [
            g1_min(x),
            g1_max(x),
            g2_min(x),
            g2_max(x),
            g3(x),
            g4(x),
            g5(x),
            g6(x),
        ]
    )


constraints = (
    {"type": "ineq", "fun": g1_min},
    {"type": "ineq", "fun": g1_max},
    {"type": "ineq", "fun": g2_min},
    {"type": "ineq", "fun": g2_max},
    {"type": "ineq", "fun": g3},
    {"type": "ineq", "fun": g4},
    {"type": "ineq", "fun": g5},
    {"type": "eq", "fun": g6},
)

bounds = optimization.helper.generate_bounds(base_oils_counts, treat_rates)
tolerance = 1e-6
alpha = 1.0


# Sketch of the algorithm
def lagrange(f_grad, x, lambda_i):
    return f_grad(x) - np.sum(lambda_i * g(x))


def nabla_g(constraints, x):
    return np.array(
        [approx_derivative(constraint["fun"], x) for constraint in constraints]
    )


def hessian_B(x, lambda_, gradient_constraints):
    n = len(x)  # Number of decision variables
    B = np.zeros((n, n))

    # Compute the second-order partial derivatives of L with respect to x_i and x_j
    for i in range(n):
        for j in range(n):
            # Calculate the (i, j) element of B using the second-order partial derivatives
            B[i, j] = -np.sum(
                lambda_ * gradient_constraints[:, i] * gradient_constraints[:, j]
            )

    return B


def l_grad(x, lambda_, gradient_objective, gradient_constraints):
    grad_L = gradient_objective(x)
    for i in range(len(lambda_)):
        grad_L -= lambda_[i] * gradient_constraints[i]
    return grad_L


def line_search(f, f_grad, x, d, alpha=1.0, rho=0.5, c=0.5):
    """
    Line search to find a suitable step size alpha for optimization.

    Args:
        f: Objective function.
        x: Current point.
        d: Search direction.
        alpha: Initial step size.
        rho: Factor for step size reduction (0 < rho < 1).
        c: Sufficient decrease parameter (0 < c < 1).

    Returns:
        alpha: Updated step size.
    """
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(d, f_grad(x)):
        alpha *= rho
    return alpha


def penalty_function(alpha, x_k, d_k, rho):
    trial_point = x_k + alpha * d_k
    constraints = g(trial_point)
    penalty = f(trial_point)

    for i, constraint in enumerate(constraints):
        penalty += rho[i] * min(0, constraint)

    return penalty


def find_step_size(x_k, d_k, rho, max_iterations=100, epsilon=1e-6):
    alpha = 1.0  # Initial step size
    for iteration in range(max_iterations):
        penalty_at_x = penalty_function(alpha, x_k, d_k, rho)
        penalty_at_x_plus_alpha = penalty_function(alpha + epsilon, x_k, d_k, rho)
        if penalty_at_x_plus_alpha < penalty_at_x:
            return alpha

        alpha /= 2.0  # Reduce the step size by half

        if alpha < epsilon:
            break

    return alpha


def calculate_step_size(x_k, mu):
    alpha = 1.0  # Default step size for points near the optimum
    m = len(constraints)
    rho_alt = np.zeros(m)
    for i in range(m):  # Iterate through each constraint
        g_i = constraints[i]["fun"]
        rho_dash = rho_alt[i]
        mu_i = mu[i]

        # Update the penalty parameter rho_alt_i
        rho_alt[i] = max(0.5 * (rho_dash + abs(mu_i)), abs(mu_i))

        # Calculate step size using the updated penalty parameter
        if g_i(x_k)[i] <= mu_i / rho_alt[i]:
            alpha = min(alpha, 1.0)
        else:
            alpha = min(alpha, 0.5 * abs(mu_i) / (abs(rho_alt[i]) * abs(g_i(x_k)[i])))

    return alpha


def solve(f, x0, constraints, bounds, max_iter=1000, eps=1e-6):
    new_bounds = optimization.helper.split_bounds(bounds)
    cons = optimization._construct.new_constraints(constraints, new_bounds)
    x = optimization.helper.clip_x(x0, new_bounds)

    sf = optimization._construct.scalar_function(f, x)

    f = optimization.helper.clip_fun(sf.fun, new_bounds)
    f_grad = optimization.helper.clip_fun(sf.grad, new_bounds)
    iteration = 0

    # lambda_i = np.array([1.0] * (len(constraints)))
    lambda_i = np.array(g(x))
    g_grad = nabla_g(constraints, x)
    B = hessian_B(x, lambda_i, g_grad)
    L = lagrange(f_grad, x, lambda_i)
    d = np.linalg.solve(B, -L)
    alpha = line_search(f, f_grad, x, d)
    alpha2 = find_step_size(x, d, lambda_i)
    iteration = 0
    x_min = x
    alpha = 1.0
    while iteration < max_iter:
        print("x", x)
        print("f(x)", f(x))
        # print("sum", sum(x))
        lambda_i = np.array(g(x))
        g_grad = nabla_g(constraints, x)
        B = hessian_B(x, lambda_i, g_grad)
        L = lagrange(f_grad, x, lambda_i)
        d = np.linalg.solve(B, -L)
        alpha = line_search(f, f_grad, x, d)
        alpha2 = find_step_size(x, d, lambda_i)
        # print("F grad :", f_grad(x))
        # print("G grad :", g_grad)
        # print("B :", B)
        # print("L :", L)
        # print("d :", d)
        # print("step :", alpha)
        # print("step :", alpha2)

        print("x :", x + alpha * d)
        # print("x :", x + alpha2 * d)

        # print("f(x) :", f(x + alpha * d))
        # print("f(x) :", f(x + alpha2 * d))

        # print("sum :", sum(x + alpha * d))
        # print("sum :", sum(x + alpha2 * d))
        x_new = x + alpha * d
        x_new = optimization.helper.clip_x(x_new, new_bounds)
        x_new /= np.sum(x_new)
        is_all_constraints_satisfied = all(
            c["fun"](x_new) >= 0 if c["type"] == "ineq" else c["fun"](x_new) == 0
            for c in constraints
        )

        obj_change = np.abs(f(x_min) - f(x_new)) and np.abs(f(x_new) - f(x))
        var_change = np.max(np.abs(x_new - x))
        if f(x_new) < f(x_min) and is_all_constraints_satisfied:
            x_min = x_new
        if (
            obj_change < tolerance
            and var_change < tolerance
            # and is_all_constraints_satisfied
        ):
            break
        # print(np.abs(f(x_new) - f(x)))
        # print("alpha:", "{:.10f}".format(alpha))
        x = x_new
        iteration += 1
    return {
        "x": x,
        "fun": f(x_min),
        "nit": iteration,
    }


manual = solve(f, x0, constraints, bounds)

scipy = minimize(
    f,
    x0,
    method="SLSQP",
    jac="2-point",
    bounds=bounds,
    constraints=constraints,
    tol=tolerance,
)
# if np.allclose(manual["x"], scipy.x):
print(" x0 :", ["{:.10f}".format(x) for x in x0])
print(" Is x close:", np.allclose(manual["x"], scipy.x))
print(" Difference:", ["{:.10f}".format(x) for x in np.abs(manual["x"] - scipy.x)])
print()
print(" Manual:", ["{:.10f}".format(x) for x in manual["x"]])
print(" Manual (fx):", manual["fun"])
print(" Manual i:", manual["nit"])
print(" Manual (sum):", sum(manual["x"]))
print()
print(" Scipy:", ["{:.10f}".format(x) for x in scipy.x])
print(" Scipy (fx):", scipy.fun)
print(" Scipy i:", scipy.nit)
print(" Scipy (sum):", sum(scipy.x))


print(np.array(x0) + 1 * np.array([0.00636293, -0.00652803]))
