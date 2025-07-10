cimport numpy as np
import numpy as np

from pymoo.model.selection import Selection


class RandomSelection(Selection):

    def _do(self, pop, int n_select, int n_parents, **kwargs):
        cdef int i
        # # number of random individuals needed
        # n_random = n_select * n_parents
        #
        # # number of permutations needed
        # n_perms = math.ceil(n_random / len(pop))
        #
        # # get random permutations and reshape them
        # P = random_permuations(n_perms, len(pop))[:n_random]
        #
        # return np.reshape(P, (n_select, n_parents))

        cdef np.ndarray P = np.ndarray((n_select, n_parents), dtype=int)
        for i in range(n_select):
            P[i, :] = np.random.choice(range(len(pop)), n_parents, replace=False)

        return P
