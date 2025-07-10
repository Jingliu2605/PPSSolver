import numpy as np

def crossover_mask(x, m):
    # convert input to output by flatting along the first axis
    _X = np.copy(x)
    _X[0][m] = x[1][m]
    _X[1][m] = x[0][m]

    return _X
