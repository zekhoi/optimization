import numpy as np

import warnings
import random


def array_to_scallar(x):
    return x.item() if isinstance(x, np.ndarray) else x


def split_bounds(bounds):
    lower_bounds, upper_bounds = zip(*bounds)

    lower_bounds = np.array(
        [float(array_to_scallar(x)) if x is not None else -np.inf for x in lower_bounds]
    )
    upper_bounds = np.array(
        [float(array_to_scallar(x)) if x is not None else np.inf for x in upper_bounds]
    )

    return lower_bounds, upper_bounds


def clip_fun(func, bounds):
    return lambda x: func(clip_x(x, bounds))


def clip_x(x, bounds):
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn(
            "Values in x were outside bounds during!",
            RuntimeWarning,
        )
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x


def generate_x(base_oil_counts, treat_rates=[]):
    return [
        1 / base_oil_counts * (1.0 - sum(treat_rates)) for _ in range(base_oil_counts)
    ] + treat_rates


def generate_bounds(base_oil_counts, treat_rates=[]):
    # [(treat_rate,1)] for treat_rates and [(0,1)] for base_oils
    return [(0.000001, 1)] * base_oil_counts + [
        (treat_rate, 1) for treat_rate in treat_rates
    ]


def generate_prices(base_oil_counts, treat_rates=[]):
    return [random.uniform(0.5, 1.5) for _ in range(base_oil_counts)] + [
        random.uniform(5, 10) for _ in range(len(treat_rates))
    ]


def generate_points(base_oils_counts, treat_rates):
    return [
        random.uniform(0.5, 1.5) for _ in range(base_oils_counts + len(treat_rates))
    ]
