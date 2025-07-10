import numpy as np
import scipy.stats as dists

from problem.enums import ValueFunction

def cost_value(total_cost, duration, **kwargs):
    return total_cost

def random_cost_value(total_cost, duration, **kwargs):
    factor = kwargs["factor"] if "factor" in kwargs else 2
    min_factor = kwargs["min_factor"] if "min_factor" in kwargs else 0.1
    return (np.random.random() * (factor - min_factor) + min_factor) * total_cost

def random_cost_dur_value(total_cost, duration, **kwargs):
    value_dist = kwargs["value_dist"] if "value_dist" in kwargs else dists.randint(1, 5)
    factor = kwargs["factor"] if "factor" in kwargs else 2

    return np.random.random() * total_cost * factor + sum(value_dist.rvs(duration - 1))

def random(total_cost, duration, **kwargs):
    factor = kwargs["factor"] if "factor" in kwargs else 1000
    min_factor = kwargs["min_factor"] if "min_factor" in kwargs else 1
    return (np.random.random() * (factor - min_factor)) + min_factor

def get_value_from_enum(value_function, total_cost, duration, **kwargs):
    if value_function is ValueFunction.COST:
        func = cost_value
    elif value_function is ValueFunction.RANDOM_COST:
        func = random_cost_value
    elif value_function is ValueFunction.COST_DUR:
        func = random_cost_dur_value
    elif value_function is ValueFunction.RANDOM:
        func = random

    return func(total_cost, duration, **kwargs)
