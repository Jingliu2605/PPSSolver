from enum import Enum

cimport numpy as np
import numpy as np

from pymoo.model.selection import Selection


class TournamentSelection(Selection):
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, func_comp=None, pressure=2):
        """

        Parameters
        ----------
        func_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.

        pressure: int
            The selection pressure to bie applied. Default it is a binary tournament.
        """

        # selection pressure to be applied
        self.pressure = pressure

        self.f_comp = func_comp
        if self.f_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

    def _do(self, pop, int n_select, int n_parents=1, **kwargs):
        # number of random individuals needed
        # n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        # n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        #P = random_permuations(n_perms, len(pop))[:n_random]
        #P = np.reshape(P, (n_select * n_parents, self.pressure))

        cdef int i
        #TODO: P doesn't work if defined as np.int_t[:, :]
        cdef np.ndarray[np.int_t, ndim=2] P = np.ndarray((n_select * n_parents, self.pressure), dtype=int)
        for i in range(n_select * n_parents):
            P[i, :] = np.random.choice(range(len(pop)), self.pressure, replace=False)

        # compare using tournament function
        S = self.f_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    if method is Ordering.Maximization:
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    elif method is Ordering.Minimization:
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    else:
        raise Exception("Unknown method.")

def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method=Ordering.Minimization, return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F, method=Ordering.Minimization, return_random_if_equal=True)

    return S[:, None].astype(int)


class Ordering(Enum):
    Minimization = 'smaller_is_better'
    Maximization = 'larger_is_better'
