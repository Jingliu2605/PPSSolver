import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class RandomEarliestSolver(HeuristicSolver):
    """
    Generate a random, feasible solution by scheduling as early as possible. This should front-load projects

    For each time step:
        For each remaining project in a random order:
            Add project at earliest feasible time step, if possible
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Random Earliest"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = np.random.permutation(range(0, self.instance.num_projects))

        for index in indices:
            earliest = self.find_earliest(index)
            if earliest > 0:
                self.add_to_portfolio(index, earliest)

        return self.portfolio
