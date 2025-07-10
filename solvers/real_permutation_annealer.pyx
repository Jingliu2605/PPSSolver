import numpy as np
cimport numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_permutation
from solvers.simulated_annealing import SimulatedAnnealing


class RealPermutationAnnealer(SimulatedAnnealing):
    """
    Simulated annealing using permutation representation
    """
    def __init__(self, initial_state, instance, random_seed):
        self.instance = instance
        super(RealPermutationAnnealer, self).__init__(initial_state, random_seed=random_seed)
        self.updates = self.steps / 100

    def move(self):
        """ Randomly swap two elements. """
        a, b = np.random.randint(0, self.state.shape[0], 2)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculates the total value of the portfolio, returning a positive value if there are violations."""
        cdef np.int_t[:] phenotype = np.argsort(self.state).astype(int)
        portfolio = build_from_permutation(phenotype, self.instance, SchedulingOrder.EARLIEST)

        return -portfolio.value
