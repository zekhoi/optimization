import numpy as np


def constraints(x, constraints):
    if constraints["eq"]:
        c_eq = np.concatenate(
            [np.atleast_1d(constraint["fun"](x)) for constraint in constraints["eq"]]
        )
    else:
        c_eq = np.zeros(0)

    if constraints["ineq"]:
        c_ieq = np.concatenate(
            [np.atleast_1d(constraint["fun"](x)) for constraint in constraints["ineq"]]
        )
    else:
        c_ieq = np.zeros(0)

    # matrix of constraints
    c = np.concatenate((c_eq, c_ieq))
    return c


def constrains_normals(x, cons, la, n, m, meq, mieq):
    # Compute the normals of the constraints
    if cons["eq"]:
        a_eq = np.vstack([con["jac"](x, *con["args"]) for con in cons["eq"]])
    else:  # no equality constraint
        a_eq = np.zeros((meq, n))

    if cons["ineq"]:
        a_ieq = np.vstack([con["jac"](x, *con["args"]) for con in cons["ineq"]])
    else:  # no inequality constraint
        a_ieq = np.zeros((mieq, n))

    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = np.zeros((la, n))
    else:
        a = np.vstack((a_eq, a_ieq))
    a = np.concatenate((a, np.zeros([la, 1])), 1)

    return a
