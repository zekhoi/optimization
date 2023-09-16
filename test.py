import optimization
import numpy as np

# Define constants and initial values
base_oils_counts = 10

treat_rates = [
    0.01,
    0.421,
]
x0 = optimization.helper.generate_x(base_oils_counts)

x0_with_additive = optimization.helper.generate_x(base_oils_counts, treat_rates)
prices = optimization.helper.generate_prices(base_oils_counts, treat_rates)
# prices = [0.49, 0.51, 1.31, 0.98, 5.2, 4.7]
flash_point = optimization.helper.generate_points(base_oils_counts, treat_rates)
pour_point = optimization.helper.generate_points(base_oils_counts, treat_rates)
FP_min = 0
PP_max = 100

# Define objective function, constraints functions and bounds


def f(x):
    # or sum(x[i] * prices[i] for i in range(len(x)))
    # or np.sum(x * prices)
    return np.dot(x, prices)


def g4(x):
    return (
        sum(
            x[i] * 51708 * np.exp((np.log(flash_point[i]) - 2.6287) ** 2 / (-0.91725))
            for i in range(len(x))
        )
        - FP_min
    )


def g5(x):
    return PP_max - sum(x[i] * 3262000 * (pour_point[i] / 1000) for i in range(len(x)))


def g6(x):
    return sum(x) - 1


def callback_function(xk):
    print("Iteration x:", ["{:.10f}".format(x) for x in xk])
    print("Total:", sum(xk))


constraints = (
    {"type": "ineq", "fun": g4},
    {"type": "ineq", "fun": g5},
    {"type": "eq", "fun": g6},
)

bounds = optimization.helper.generate_bounds(base_oils_counts, treat_rates)
tolerance = 1e-6
alpha = 1.0

# Sketch of the algorithm


def find_direction(f, x0, constraints, bounds, max_iter=1000, eps=1e-6):
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

    jac_eq = optimization._construct.constraint_gradients(cons["eq"], x)
    jac_ineq = optimization._construct.constraint_gradients(cons["ineq"], x)
    jac_constraints = np.concatenate((jac_eq, jac_ineq), axis=0)
    constraints_satisfied = False
    iter_count = 0

    print("Iteration:", iter_count)
    print("Constraints satisfied:", constraints_satisfied)
    print("Ratio:", sum(x))
    print("X:", ["{:.10f}".format(x) for x in x])
    print("Constraints g4 (must lower):", constraints[0]["fun"](x))
    print("Constraints g5 (must greater):", constraints[1]["fun"](x))
    print("Constraints g6 (must 0):", constraints[2]["fun"](x))
    while (
        not constraints_satisfied
        and np.linalg.norm(jac_constraints) > eps
        and iter_count < max_iter
    ):
        iter_count += 1
        print("Iteration:", iter_count)
        print("Constraints satisfied:", constraints_satisfied)
        print("Ratio:", sum(x))
        print("X:", ["{:.10f}".format(x) for x in x])
        print("Constraints g4 (must lower):", constraints[0]["fun"](x))
        print("Constraints g5 (must greater):", constraints[1]["fun"](x))
        print("Constraints g6 (must 0):", constraints[2]["fun"](x))
        jac_eq = optimization._construct.constraint_gradients(cons["eq"], x)
        jac_ineq = optimization._construct.constraint_gradients(cons["ineq"], x)
        jac_constraints = np.concatenate((jac_eq, jac_ineq), axis=0)
        print(jac_constraints)
        A = np.vstack((f_grad(x), jac_constraints))
        b = np.zeros(A.shape[0])
        b[0] = -1  # Minimize the objective function
        d1 = np.linalg.lstsq(A, b, rcond=None)[0]
        # d2 = optimization._solver.find_direction(jac_constraints, f_grad(x))
        # d3 = -f_grad(x) - np.sum(jac_constraints, axis=0)
        d4 = optimization._compute.lagrange(
            f(x),
            jac_constraints,
            np.array([0.9 for x in range(len(jac_constraints))]),
        )
        # step1 = optimization._solver.find_step_size(f, x, constraints, d1)
        # step2 = optimization._solver.find_step_size(f, x, constraints, d2)
        # step3 = optimization._solver.find_step_size(f, x, constraints, d3)
        step4 = optimization._solver.find_step_size(f, x, constraints, d4)

        # x_new1 = optimization.helper.clip_x(x + step1 * d1, new_bounds)
        # x_new2 = optimization.helper.clip_x(x + step2 * d2, new_bounds)
        # x_new3 = optimization.helper.clip_x(x + step3 * d3, new_bounds)
        x_new4 = optimization.helper.clip_x(x + step4 * d4, new_bounds)

        # if f(x_new1) < f(x):
        #     print("x1 :", ["{:.10f}".format(x) for x in x_new1])
        #     print("x1 sum :", sum(x_new1))
        #     print("fx1:", f(x_new1))
        #     x = x_new1
        # if f(x_new2) < f(x):
        #     print("x1 :", ["{:.10f}".format(x) for x in x_new2])
        #     print("x1 sum :", sum(x_new2))
        #     print("fx1:", f(x_new2))
        #     x = x_new2
        # if f(x_new3) < f(x):
        #     print("x3 :", ["{:.10f}".format(x) for x in x_new3])
        #     print("x3 sum :", sum(x_new3))
        #     print("fx3:", f(x_new3))
        #     x = x_new3
        # if f(x_new4) < f(x):
        #     print("x4 :", ["{:.10f}".format(x) for x in x_new4])
        #     print("x4 sum :", sum(x_new4))
        #     print("fx4:", f(x_new4))
        #     x = x_new4

        print("Direction:", d4)
        print("Step:", step4)
        print()
        constraints_satisfied = all(
            np.all(constraint["fun"](x) >= 0)
            if constraint["type"] == "ineq"
            else np.isclose(constraint["fun"](x), 0)
            for constraint in constraints
        )

        # print("Testing :", x_new)

        # if np.linalg.norm(x_new - x) < eps:
        #     break

        # # Update x for the next iteration
        # x = x_new


find_direction(f, x0_with_additive, constraints, bounds)
