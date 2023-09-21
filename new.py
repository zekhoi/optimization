import optimization
import numpy as np
import cvxpy as cp
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import minimize
from optimization.table import d2270
from scipy.interpolate import interp1d


# Define constants and initial values
base_oils_counts = 2

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
v40 = [111, 289]
v100 = [297, 523]

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


def g3_base(vB_40, vB_100):
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
        N = (np.log10(H) - np.log10(U)) / np.log(Y)
        return (np.exp(N) - 1) / 0.00715 + 100
    elif U == H:
        return 100


def g1_min(x):
    return kv(40)(x) - v40_min


def g1_max(x):
    return v40_max - kv(40)(x)


def g2_min(x):
    return kv(100)(x) - v100_min


def g2_max(x):
    return v100_max - kv(100)(x)


def g3(x):
    return g3_base(kv(40)(x), kv(100)(x))


def g4(x):
    fp_in_fahrenheit_blended = sum(
        x[i]
        * 51708
        * np.exp(
            (
                (
                    np.log(optimization.helper.celcius_to_fahrenheit(flash_point[i]))
                    - 2.6287
                )
                ** 2
            )
            / (-0.91725)
        )
        for i in range(len(x))
    )
    fp_in_fahrenheit = np.exp(
        np.power((-0.91725 * np.log(fp_in_fahrenheit_blended / 51708)), 0.5) + 2.6827
    )
    fp_in_celcius = optimization.helper.fahrenheit_to_celcius(fp_in_fahrenheit)
    return fp_in_celcius - FP_min  # - FP_min


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


def g6(x):
    return sum(x) - 1


def g(x):
    return np.array(
        [g1_min(x), g1_max(x), g2_min(x), g2_max(x), g3(x), g4(x), g5(x), g6(x)]
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


def solve(f, x0, constraints, bounds, max_iter=1000, eps=1e-6):
    new_bounds = optimization.helper.split_bounds(bounds)
    cons = optimization._construct.new_constraints(constraints, new_bounds)
    x = optimization.helper.clip_x(x0, new_bounds)

    sf = optimization._construct.scalar_function(f, x)

    f = optimization.helper.clip_fun(sf.fun, new_bounds)
    f_grad = optimization.helper.clip_fun(sf.grad, new_bounds)

    is_all_constraints_satisfied = False
    iteration = 0
    x_min = x
    alpha = 1.0
    print("Objective:", f(x))
    print("Gradient:", f_grad(x))
    g_grad = np.array(
        [optimization._compute.constraint_gradient(c["fun"], x) for c in constraints]
    )
    gradient = optimization._compute.objective_gradient(f, x)
    jacobian = optimization._construct.jacobian(constraints, x)
    print("Jacobian:", jacobian)
    g_grad = np.array(
        [optimization._compute.constraint_gradient(c["fun"], x) for c in constraints]
    )
    H = np.array(
        approx_derivative(f_grad, x, rel_step=1e-6, f0=gradient),
    )
    print("H:", H)
    A = jacobian
    print("A:", A)
    b = np.array(jacobian)
    lambda_ = np.maximum(0, g(x))
    print("lambda:", lambda_)
    print(-np.diag(lambda_))
    kkt_matrix = np.vstack(
        [
            np.hstack([H, A.T]),
            np.hstack([A, -np.diag(lambda_)]),
        ]
    )
    if np.linalg.det(kkt_matrix) == 0:
        # print("Singular matrix")
        kkt_matrix += np.eye(len(kkt_matrix)) * eps
    print("kkt_matrix:", kkt_matrix)
    rhs = np.concatenate([gradient, g(x)])
    print("rhs:", rhs)
    # check singularity matrix
    # print("kkt_matrix:", kkt_matrix)
    # print("rhs:", rhs)
    direction = -np.linalg.solve(kkt_matrix, rhs)
    print("direction:", direction)

    # lagrange_gradient = gradient - np.sum(lambda_ * jacobian, axis=0)
    # print("Lagrange:", lagrange_gradient)
    # print("g_grad:", g_grad)
    # print("b", b)
    while iteration < max_iter:
        # gradient = optimization._compute.objective_gradient(f, x)
        gradient = f_grad(x)
        jacobian = optimization._construct.jacobian(constraints, x)
        g_grad = np.array(
            [
                optimization._compute.constraint_gradient(c["fun"], x)
                for c in constraints
            ]
        )
        H = np.array(
            approx_derivative(f_grad, x, rel_step=1e-6, f0=gradient),
        )
        A = jacobian
        b = np.array(jacobian)
        lambda_ = np.maximum(0, g(x))
        kkt_matrix = np.vstack(
            [
                np.hstack([H, A.T]),
                np.hstack([A, -np.diag(lambda_)]),
            ]
        )
        rhs = np.concatenate([gradient, g(x)])
        # check singularity matrix
        if np.linalg.det(kkt_matrix) == 0:
            # print("Singular matrix")
            kkt_matrix += np.eye(len(kkt_matrix)) * eps
        # print("kkt_matrix:", kkt_matrix)
        # print("rhs:", rhs)
        direction = -np.linalg.solve(kkt_matrix, rhs)

        # lagrange_gradient = gradient - np.sum(lambda_ * jacobian, axis=0)
        # print("Lagrange:", lagrange_gradient)
        # print("Objective:", f(x))
        # print("Gradient:", gradient)
        # print("g_grad:", g_grad)
        # print("Jacobian:", jacobian)
        # print("H:", H)
        # print("A:", A)
        # print("b", b)
        # print("kkt_matrix:", kkt_matrix)
        # print("rhs:", rhs)
        # print("lambda:", lambda_)
        # print("direction:", direction)
        # if iteration == 10:
        #     return {
        #         "x": x,
        #         "fun": f(x),
        #     }
        # Update x and the Lagrange multiplier
        x_new = x + alpha * direction[: len(x)]
        x_new = optimization.helper.clip_x(x_new, new_bounds)
        x_new /= np.sum(x_new)
        # print("x_new:", ["{:.10f}".format(x) for x in x_new])
        # print("x:", ["{:.10f}".format(x) for x in x])
        # check constraints satisfaction
        # for i in range(len(constraints)):
        # print("Constraint value x_new:", constraints[i]["fun"](x_new))
        # print("Constraint value x:", constraints[i]["fun"](x))
        # if (
        #     constraints[i]["fun"](x_new) < 0
        #     if constraints[i]["type"] == "ineq"
        #     else constraints[i]["fun"](x_new) > 0
        # ):
        #     print("Constraint", i, "not satisfied")
        # else:
        #     print("Constraint", i, "satisfied")

        is_all_constraints_satisfied = all(
            c["fun"](x_new) >= 0 if c["type"] == "ineq" else c["fun"](x_new) == 0
            for c in constraints
        )
        # print("all constraints satisfied:", is_all_constraints_satisfied)
        # print("fx_new:", f(x_new))
        # print("fx:", f(x))
        # print()

        obj_change = np.abs(f(x_min) - f(x_new)) and np.abs(f(x_new) - f(x))
        var_change = np.max(np.abs(x_new - x))
        # Check for convergence
        # print("obj_change:", obj_change < tolerance, obj_change)
        # print("var_change:", var_change < tolerance, var_change)
        # print("is_all_constraints_satisfied:", is_all_constraints_satisfied)
        # print()
        if f(x_new) < f(x_min) and is_all_constraints_satisfied:
            x_min = x_new
        if (
            obj_change < tolerance
            and var_change < tolerance
            # and is_all_constraints_satisfied
        ):
            break
        # print(np.abs(f(x_new) - f(x)))
        alpha = alpha * 0.5 if np.abs(f(x_new) - f(x)) < tolerance else alpha / 0.5
        # print("alpha:", "{:.10f}".format(alpha))
        x = x_new
        iteration += 1
    return {
        "x": x_min,
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
