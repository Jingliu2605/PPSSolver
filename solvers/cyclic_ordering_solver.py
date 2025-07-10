from abc import abstractmethod

import numpy as np

from problem.enums import PortfolioOrdering
from solvers.heuristic_solver import HeuristicSolver


class CyclicOrderingSolver(HeuristicSolver):
    """
    Generate a feasible solution using cyclic project selection.

    While not completed:
        For each time step:
            Select feasible project by preference (if available)
            Continue to next time step
        If no projects selected, mark completed
    """

    def __init__(self, instance, ordering=PortfolioOrdering.VALUE_DESC, seed=1):
        self.ordering = ordering
        super().__init__(instance, seed)
        self.name = f"Cyclic {ordering.value}"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = self._preferences(self.ordering)

        done = False

        while not done:
            done = True  # return true if no new project was added in this iteration
            # cycle through time steps and add a project, if possible
            for t in range(1, self.instance.planning_window + 1):
                num_elements = len(indices)
                for i in range(num_elements):
                    candidate_p = int(indices[i])
                    if self.feasibility_check(candidate_p, t):
                        self.add_to_portfolio(candidate_p, t)
                        indices = np.delete(indices, i)
                        done = False
                        break  # move to next time step

        return self.portfolio

    def _preferences(self, ordering):
        pref = np.zeros(self.instance.num_projects)
        if ordering is PortfolioOrdering.COST_ASC:  # cyclic low cost
            pref = np.fromiter((p.total_cost for p in self.instance.projects), dtype=np.double)
        elif ordering is PortfolioOrdering.COST_DESC:  # cyclic high cost
            pref = np.fromiter((-p.total_cost for p in self.instance.projects), dtype=np.double)
        elif ordering is PortfolioOrdering.VALUE_ASC:  # cyclic low value
            pref = np.fromiter((np.sum(p.value) for p in self.instance.projects), dtype=np.double)
        elif ordering is PortfolioOrdering.VALUE_DESC:  # cyclic high value
            pref = np.fromiter((-np.sum(p.value) for p in self.instance.projects), dtype=np.double)
        elif ordering is PortfolioOrdering.CV_RATIO:  # value / cost ratio
            pref = np.fromiter((p.total_cost / np.sum(p.value) for p in self.instance.projects), dtype=np.double)

        return pref.argsort()
