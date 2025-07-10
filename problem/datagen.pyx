from itertools import chain
from math import exp, floor
from warnings import warn

import numpy as np

def mvlnorm_generate_costdur(num_samples, log_mean_dur=2.191054, log_var_dur=0.246245, log_mean_cost=6.642006,
                             log_var_cost=1.555780, cov=0.374572):
    """
    Generate data using a multi-variate log-normal distribution. The default values for means, variances,
    and covariance are derived from the (log) data extracted from the 2016 IIP. The MV log-normal distribution is
    simulated by using a MV normal distribution and exponentiating the resulting output.

    The multivariate normal distribution is generated using the log_mean values and a covariance matrix specified by
    [log_var_dur, cov]
    [cov, log_var_cost]

    :param log_mean_dur: Mean of the log-scaled project durations
    :param log_var_dur: Variance of the log-scaled project durations
    :param log_mean_cost: Mean of the log-scaled total project costs
    :param log_var_cost: Variance of the log-scaled total project costs
    :param cov: Covariance between log-scaled project durations and costs
    :param num_samples: The number of duration, cost samples to generate
    :return: A (num_samples x 2) numpy array of duration, cost samples rounded to the nearest integer
    """
    cov_mat = [[log_var_dur, cov], [cov, log_var_cost]]

    data = np.random.multivariate_normal([log_mean_dur, log_mean_cost], cov_mat, num_samples)
    data = np.rint(np.exp(data)).astype(int)

    return data

def distribute_uniform(total, n):
    """
    Method to decompose total into n uniformly distributed integers
    :param total:
    :param n:
    :return:
    """

    values = np.random.choice(range(total), n - 1, replace=False)
    sorted_values = np.sort(values)
    result = np.zeros(n, dtype=np.double)

    result[0] = sorted_values[0]

    for i in range(1, n - 1):
        result[i] = sorted_values[i] - sorted_values[i - 1]

    result[n - 1] = total - sorted_values[n - 2]

    return result

def even_cost_per_year(cost, years):
    """
    Distribute the cost as evenly ass possible over all years. Any remainder is allocated to the later years
    :param cost: The total project cost
    :param years: The number of years
    :return: An array of per-year costs, as evenly distributed as possible
    """

    # use integer division to get base amount
    base_year = cost // years
    leftover = cost % years

    result = np.ones(years, dtype=np.double) * base_year
    if leftover > 0:
        result[-leftover::] += 1

    return result

def ramped_cost_per_year(cost, years):
    """
    Generate a ramped cost distribution over time using a uniformly distributed total cost. This is done by first
    uniformly distributing the costs using the distribute_uniform function. Then, the distributed costs are sorted
    and every other sample is taken, first in a forward pass and then a reverse pass.

    Example: assume cost=20, years=4 and distribute_uniform returned [7, 3, 6, 4]. This would be sorted as [3, 4, 6,
    7] and sampled as [3, 6, 7, 4].

    For an odd number of years, such as cost=20, years=5, with [6, 2, 4, 5, 3], this would be sorted as [2, 3, 4, 5,
    6] and sampled as [2, 4, 6, 5, 3]

    :param cost: The total project cost
    :param years: The number of years the project will run
    :return: A ramped distribution of costs per year
    """
    costs = distribute_uniform(cost, years)
    sorted_values = np.sort(costs)

    # determine the start value for the second half, depending on whether there is an even or odd number of components
    ramp_down = years - 1 if years % 2 == 0 else years - 2

    result = np.zeros(years, dtype=np.double)
    index = 0
    for i in chain(range(0, years, 2), range(ramp_down, 0, -2)):
        result[index] = sorted_values[i]
        index += 1

    return result

def fuzzy_weibull_cost_distribution(total_cost, duration, shape=1.589, scale=0.71, std_shape=2, std_scale=0.3):
    fuzzy_shape = np.random.normal(shape, std_shape)
    while fuzzy_shape < 0.01:
        fuzzy_shape = np.random.normal(shape, std_shape)

    fuzzy_scale = max(np.random.normal(scale, std_scale), 0.1)

    dist = np.zeros(duration, dtype=np.double)
    costs = np.zeros(duration, dtype=np.double)

    cumulative_cost = 0
    prev_dist = 0
    for t in range(duration - 1):
        dist[t] = _weibull_estimate((t + 1) / duration, fuzzy_shape, fuzzy_scale)
        cost = floor((dist[t] - prev_dist) * total_cost)
        costs[t] = cost
        cumulative_cost += cost
        prev_dist = dist[t]

    dist[duration - 1] = 1
    costs[duration - 1] = total_cost - cumulative_cost

    return costs

def _weibull_estimate(time, shape=1.589, scale=0.71):
    if time < 0:
        warn("time < 0 in _weibull_estimate. Returning 0.")
        return 0
    elif time >= 1:
        return 1

    num = 1 - exp(-(time / scale) ** shape)
    denom = 1 - exp(-(1 / scale) ** shape)

    return num / denom
