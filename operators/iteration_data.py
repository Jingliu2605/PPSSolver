import numpy as np


class IterationData:

    def __init__(self, iteration, best, mean, std, iter_min, iter_max, best_sol):
        self.iter = iteration
        self.best = best
        self.mean = mean
        self.std = std
        self.min = iter_min
        self.max = iter_max
        self.best_sol = best_sol

    def __str__(self):
        sol_str = np.array2string(self.best_sol).replace('\n', '')
        return f"{self.iter}, {self.best}, {self.mean}, {self.std}, {self.min}, {self.max}, {sol_str}"
