import numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_permutation
from solvers.simulated_annealing import SimulatedAnnealing


class PermutationAnnealer(SimulatedAnnealing):
    """
    Simulated annealing using permutation representation
    """
    def __init__(self, initial_state, instance, random_seed):
        self.instance = instance
        super(PermutationAnnealer, self).__init__(initial_state, random_seed=random_seed)
        self.updates = self.steps / 100

    def move(self):
        """ Swaps the order of two randomly selected projects. """
        a, b = np.random.randint(0, self.state.shape[0], 2)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """Calculates the total value of the portfolio, returning a positive value if there are violations."""
        portfolio = build_from_permutation(self.state, self.instance, SchedulingOrder.EARLIEST)

        return -portfolio.value
