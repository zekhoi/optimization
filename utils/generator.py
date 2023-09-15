def generate_x(base_oil_counts, treat_rates=[]):
    initial_base_oils = [
        1 / base_oil_counts * (1.0 - sum(treat_rates)) for _ in range(base_oil_counts)
    ]
    return initial_base_oils + treat_rates


def generate_bounds(base_oil_counts, treat_rates=[]):
    # [(treat_rate,1)] for treat_rates and [(0,1)] for base_oils
    return [(0.000001, 1)] * base_oil_counts + [
        (treat_rate, 1) for treat_rate in treat_rates
    ]
