# cython: profile=False
import math

import numpy as np

from pymoo.model.crossover import Crossover
from pymoo.operators.crossover.crossover_mask import crossover_mask


class HalfUniformCrossover(Crossover):

    def __init__(self, prob_hux=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_hux = prob_hux

    def _do(self, problem, X, **kwargs):
        cdef int n_matings = X.shape[1]
        cdef int n_var = X.shape[2]
        #_, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        not_equal = X[0] != X[1]

        # create for each individual the crossover range
        for i in range(n_matings):
            I = np.where(not_equal[i])[0]

            #generate a random permutation and take indices from the first half
            n = math.ceil(len(I) / 2)
            if n > 0:
                _I = I[np.random.permutation(len(I))[:n]]
                M[i, _I] = True

        _X = crossover_mask(X, M)
        return _X

    def __str__(self):
        return f"HUX (p_hux={self.prob_hux})"

    def __repr__(self):
        return self.__str__()
