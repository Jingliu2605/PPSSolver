from itertools import chain

cimport numpy as np
import numpy as np

from pymoo.model.crossover import Crossover

class OrderedCrossover(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.sentinel = kwargs["sentinel"] if "sentinel" in kwargs else -99999

    def _do(self, problem, np.int_t[:, :, :] x, **kwargs):
        """
        Based on Pyvolution
        https://github.com/inspectorG4dget/Genetic-Framework/blob/master/pyvolution/pyvolution/crossover.py
        :param problem:
        :param x:
        :param kwargs:
        :return:
        """
        cdef np.int_t[:, :, :] _X
        cdef np.int_t[:] p1, p2, c1, c2, s1, s2
        cdef int i, j, a, b, ind1, ind2, n_matings, n_var

        n_matings = x.shape[1]
        n_var = x.shape[2]

        _X = np.copy(x)
        for j in range(n_matings):
            p1 = _X[0][j]
            p2 = _X[1][j]

            c1 = np.full(n_var, self.sentinel, dtype=int)
            c2 = np.full(n_var, self.sentinel, dtype=int)

            # pick a segment to copy directly
            a, b = np.random.choice(range(n_var), 2, replace=False)
            if a > b:
                a, b = b, a

            c1[a:b] = p1[a:b]
            c2[a:b] = p2[a:b]

            s1 = np.setdiff1d(p1, c2, assume_unique=True)  # elements to insert in off2
            s2 = np.setdiff1d(p2, c1, assume_unique=True)  # elements to insert in off1

            ind1 = 0
            ind2 = 0

            # loop through other parent and insert into child in order
            # skip the copied segment by chaining the segments before and after into a single range
            for i in chain(range(a), range(b, len(c1))):
                if c1[i] == self.sentinel:
                    c1[i] = s2[ind1]
                    ind1 += 1
                if c2[i] == self.sentinel:
                    c2[i] = s1[ind2]
                    ind2 += 1

            _X[0][j] = c1
            _X[1][j] = c2

        return np.asarray(_X)

    def __str__(self):
        return f"OX (rate={self.prob})"

    def __repr__(self):
        return self.__str__()