import operator

import numpy as np


def roulette_wheel_select(candidates: np.array, weights: np.array, alpha=None):
    if alpha is None:
        # TODO: convert weights to nparray, if not already
        p = weights/weights.sum()
    else:
        weight_scaled = np.power(weights, alpha)
        p = weight_scaled / weight_scaled.sum()

    return np.random.choice(candidates, p=p)


# return a rolling window over a numpy array
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def nparray_tostring_helper(array):
    return np.array2string(array).replace('\n', '')


def display_results(solution, name, planning_window, run_time):
    print(f"{name}: {solution.value:0.3f}")

    # print(f"{solution.value:0.3f}", end=" & ")
    # print(f"\n--------------{name}--------------")
    # print(f'Value: {solution.value:.3f}')
    # print("Cost:", end=" ")
    # print(*solution["cost"], sep=", ")
    # if solution.result is not None:
    #    print(start_count(solution.result, planning_window))
    print("Time: ", f"{run_time:.3E}s")
    print()


def combine_dicts(a, b, op=operator.add):
    """Combine two dictionaries by applying op to each pair of values with the same key. Copied from here:
    https://stackoverflow.com/questions/11011756/is-there-any-pythonic-way-to-combine-two-dicts-adding-values-for
    -keys-that-appe """
    return {**a, **b, **{k: op(a[k], b[k]) for k in a.keys() & b}}


def print_projects(projects):
    for p in projects:
        print(p)
