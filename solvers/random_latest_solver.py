import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class RandomLatestSolver(HeuristicSolver):
    """
    Generate a random, feasible solution by scheduling as late as possible. This should back-load projects

    For each time step:
        For each remaining project in a random order:
            Add project at latest feasible time step, if possible
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Random Latest"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = np.random.permutation(range(0, self.instance.num_projects))

        for index in indices:
            latest = self.find_latest(index)
            if latest > 0:
                self.add_to_portfolio(index, latest)

        return self.portfolio
