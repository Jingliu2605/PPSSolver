import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class PrefEarliestSolver(HeuristicSolver):
    """
    Generate a feasible solution by selecting projects according to their value-to-cost ratio and scheduling as early as
    possible.

    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Pref Earliest"

    def solve(self):
        """
        Generate a solution
        :return: a Portfolio object
        """

        indices = self.__preferences()

        for candidate_p in indices:
            self.portfolio.add_earliest_feasible(candidate_p, self.instance.projects[candidate_p], self.instance)

        return self.portfolio

    def __preferences(self):
        pref = np.fromiter((p.total_cost / np.sum(p.value) for p in self.instance.projects), dtype=np.double)

        return pref.argsort()
