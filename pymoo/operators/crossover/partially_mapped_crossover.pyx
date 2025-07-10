from itertools import chain

cimport numpy as np
import numpy as np

from pymoo.model.crossover import Crossover


class PartiallyMappedCrossover(Crossover):
    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, np.int_t[:, :, :] X, **kwargs):
        #cdef np.ndarray[np.int_t, ndim=3] _X
        #cdef np.ndarray[np.int_t] ind1, ind2, segment1
        cdef np.int_t[:, :, :] _X
        cdef np.int_t[:] ind1, ind2, segment1
        cdef int i, j, point1, point2, val, size
        cdef dict mapping, rev_mapping

        n_matings = X.shape[1]
        n_var = X.shape[2]

        _X = np.copy(X)
        for j in range(n_matings):
            ind1 = _X[0][j]
            ind2 = _X[1][j]

            size = min(ind1.shape[0], ind2.shape[0])
            point1, point2 = np.random.choice(range(size + 1), 2, replace=False)

            if point1 > point2:
                point1, point2 = point2, point1

            mapping = dict()
            rev_mapping = dict()
            # define the mappings
            for i in range(point1, point2):
                mapping[ind2[i]] = ind1[i]
                rev_mapping[ind1[i]] = ind2[i]

            # swap between the indices using a temporary variable (numpy slices are views)
            segment1 = np.copy(ind1[point1:point2])
            ind1[point1:point2] = ind2[point1:point2]
            ind2[point1:point2] = segment1

            # perform the mapping on the
            for i in chain(range(point1), range(point2 + 1, size)):
                # forward mapping
                val = ind1[i]
                while val in mapping:
                    val = mapping[val]
                ind1[i] = val

                # backward mapping
                val = ind2[i]
                while val in rev_mapping:
                    val = rev_mapping[val]
                ind2[i] = val

            # TODO: does this need to be cloned?

            _X[0][j] = ind1
            _X[1][j] = ind2
            #_X[0][j] = ind1
            #_X[1][j] = ind2

        return np.asarray(_X)

    def __str__(self):
        return f"PMX (rate={self.prob})"

    def __repr__(self):
        return self.__str__()
