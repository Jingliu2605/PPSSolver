import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class PrefLatestSolver(HeuristicSolver):
    """
    Generate a random, feasible solution by scheduling as late as possible. This should back-load projects

    For each time step:
        For each remaining project in a random order:
            Add project at latest feasible time step, if possible
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Pref Latest"

    def solve(self):
        """
        Generate a solution
        :return: a Portfolio object
        """

        indices = self.__preferences()

        for candidate_p in indices:
            self.portfolio.add_latest_feasible(candidate_p, self.instance.projects[candidate_p], self.instance)

        return self.portfolio

    def __preferences(self):
        pref = np.fromiter((p.total_cost / np.sum(p.value) for p in self.instance.projects), dtype=np.double)

        return pref.argsort()
