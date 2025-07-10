cimport numpy as np
import numpy as np

from pymoo.model.crossover import Crossover

class SimilarJobOrderCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.sentinel = kwargs["sentinel"] if "sentinel" in kwargs else -99999

    def _do(self, problem, x, **kwargs):
        cdef np.int_t[:, :, :] _X
        cdef np.int_t[:] p1, p2, c1, c2, s1, s2
        cdef int i, j, index1, index2, cut_point

        cdef int n_matings = x.shape[1]
        cdef int n_var = x.shape[2]

        _X = np.copy(x)
        for j in range(n_matings):
            cut_point = np.random.randint(1, n_var)
            p1 = _X[0][j]
            p2 = _X[1][j]
            c1 = np.full(n_var, self.sentinel, dtype=int)
            c2 = np.full(n_var, self.sentinel, dtype=int)

            for i in range(cut_point):
                c1[i] = p1[i]
                c2[i] = p2[i]

            for i in range(cut_point, n_var):
                # TODO: block_size
                if p1[i] == p2[i]:
                    c1[i] = p1[i]
                    c2[i] = p1[i]

            s1 = np.setdiff1d(p1, c2, assume_unique=True)  # elements to insert in off2
            s2 = np.setdiff1d(p2, c1, assume_unique=True)  # elements to insert in off1

            index1 = 0
            index2 = 0
            for i in range(cut_point, n_var):
                if c1[i] == self.sentinel:
                    c1[i] = s2[index1]
                    index1 += 1
                if c2[i] == self.sentinel:
                    c2[i] = s1[index2]
                    index2 += 1

            _X[0][j] = c1
            _X[1][j] = c2

        return np.asarray(_X)

    def __str__(self):
        return f"SJOX (rate={self.prob})"

    def __repr__(self):
        return self.__str__()