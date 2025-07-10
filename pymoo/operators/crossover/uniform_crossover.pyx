import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.crossover_mask import crossover_mask


class UniformCrossover(Crossover):

    def __init__(self, prob_swap=0.5, **kwargs):
        super().__init__(2, 1, **kwargs)
        self.prob_swap = prob_swap

    def _do(self, problem, X, **kwargs):
        cdef int n_matings = X.shape[1]
        cdef int n_var = X.shape[2]

        #_X = np.copy(X)
        M = np.random.random((n_matings, n_var)) < self.prob
        _X = crossover_mask(X, M)
        return _X

    def __str__(self):
        return f"UX (p_swap={self.prob_swap})"

    def __repr__(self):
        return self.__str__()
