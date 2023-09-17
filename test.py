import optimization
import numpy as np
from scipy.optimize._numdiff import approx_derivative

# Define constants and initial values
base_oils_counts = 4

treat_rates = [
    0.01,
    0.421,
]
x0 = optimization.helper.generate_x(base_oils_counts)

x0_with_additive = optimization.helper.generate_x(base_oils_counts, treat_rates)
prices = optimization.helper.generate_prices(base_oils_counts, treat_rates)
# prices = [0.49, 0.51, 1.31, 0.98, 5.2, 4.7]
flash_point = optimization.helper.generate_flash_points(base_oils_counts, treat_rates)
pour_point = optimization.helper.generate_pour_points(base_oils_counts, treat_rates)
FP_min = 100
PP_max = 0

# Define objective function, constraints functions and bounds


def f(x):
    # or sum(x[i] * prices[i] for i in range(len(x)))
    # or np.sum(x * prices)
    return np.dot(x, prices)


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
    return fp_in_celcius - FP_min  # FP_min -


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
    return pp_in_celcius - PP_max  # - PP_max


def g6(x):
    return sum(x) - 1


def callback_function(xk):
    print("Iteration x:", ["{:.10f}".format(x) for x in xk])
    print("Total:", sum(xk))


def fconstraints(x):
    return np.array([g4(x), g5(x), g6(x)])


constraints = (
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
    # f_grad = optimization._compute.objective_gradient(f, x)
    meq = sum(map(len, [np.atleast_1d(c["fun"](x, *c["args"])) for c in cons["eq"]]))
    mieq = sum(map(len, [np.atleast_1d(c["fun"](x, *c["args"])) for c in cons["ineq"]]))
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = np.array([1, m]).max()
    # n = The number of independent variables
    n = len(x)
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (
        (3 * n1 + m) * (n1 + 1)
        + (n1 - meq + 1) * (mineq + 2)
        + 2 * mineq
        + (n1 + mineq) * (n1 - meq)
        + 2 * meq
        + n1
        + ((n + 1) * n) // 2
        + 2 * m
        + 3 * n
        + 3 * n1
        + 1
    )
    len_jw = mineq
    w = np.zeros(len_w)
    jw = np.zeros(len_jw)

    fx = f(x)
    g = np.append(f_grad(x), 0.0)
    c = optimization._evaluate.constraints(x, cons)
    a = optimization._evaluate.constrains_normals(x, cons, la, n, m, meq, mieq)

    cek = optimization._construct.constraint_gradients([constraints[1]], x)
    jac_ineq = optimization._construct.constraint_gradients(cons["ineq"], x)
    jac_eq = optimization._construct.constraint_gradients(cons["eq"], x)
    jac_constraints = np.concatenate((jac_ineq, jac_eq), axis=0)
    lagrange = f(x) - sum(fconstraints(x))
    g_grad = fconstraints(x)
    direct = np.linalg.lstsq(jac_constraints, g_grad, rcond=None)[0]
    step = optimization._solver.find_step_size(f, x, constraints, direct, new_bounds)
    constraints_satisfied = False
    iter_count = 0

    print("Iteration:", iter_count)
    print("Constraints satisfied:", constraints_satisfied)
    print("Ratio:", sum(x))
    print("f(x):", f(x))
    print("x:", ["{:.10f}".format(x) for x in x])
    print(
        "Constraints g4 (must greater than {0}):".format(0),
        constraints[0]["fun"](x),
    )
    print(
        "Constraints g5 (must lower than {0}):".format(0),
        constraints[1]["fun"](x),
    )
    print("Constraints g6 (must 0):", constraints[2]["fun"](x))
    print()
    while (
        not constraints_satisfied
        and np.linalg.norm(jac_constraints) > eps
        and iter_count < max_iter
    ):
        iter_count += 1
        jac_eq = optimization._construct.constraint_gradients(cons["eq"], x)
        jac_ineq = optimization._construct.constraint_gradients(cons["ineq"], x)
        jac_constraints = np.concatenate((jac_eq, jac_ineq), axis=0)
        d = np.linalg.lstsq(jac_constraints, g_grad, rcond=None)[0]
        step = optimization._solver.find_step_size(
            f, x, constraints, direct, new_bounds
        )
        # print("Step:", step)
        # print("Direction:", d)
        x_new = optimization.helper.clip_x(x + step * d, new_bounds)
        if f(x_new) < f(x):
            print("x :", ["{:.10f}".format(x) for x in x_new])
            print("x sum :", sum(x_new))
            print("fx:", f(x_new))

        # print("Direction:", d)
        # print("Step:", step)
        # print()
        constraints_satisfied = all(
            np.all(constraint["fun"](x) >= 0)
            if constraint["type"] == "ineq"
            else np.isclose(constraint["fun"](x), 0)
            for constraint in constraints
        )

        # print(
        #     "Constraints satisfied:",
        #     [
        #         np.all(constraint["fun"](x) >= 0)
        #         if constraint["type"] == "ineq"
        #         else np.isclose(constraint["fun"](x), 0)
        #         for constraint in constraints
        #     ],
        # )
        if np.linalg.norm(x_new - x) < eps and constraints_satisfied:
            print(
                "Optimization terminated successfully. The solution is within tolerance."
            )
            break

        # Update x for the next iteration
        x = x_new
        print("Iteration:", iter_count)
        print("Constraints satisfied:", constraints_satisfied)
        print("Ratio:", sum(x))
        print("f(x):", f(x))
        print("x:", ["{:.10f}".format(x) for x in x])
        print(
            "Constraints g4 (must greater than {0}):".format(0),
            constraints[0]["fun"](x),
        )
        print(
            "Constraints g5 (must lower than {0}):".format(0),
            constraints[1]["fun"](x),
        )
        print("Constraints g6 (must 0):", constraints[2]["fun"](x))


solve(f, x0_with_additive, constraints, bounds)
