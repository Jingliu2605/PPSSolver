import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class PrefEarliestConstraintSolver(HeuristicSolver):
    """
    Generate a random, feasible solution by scheduling as late as possible. This should back-load projects

    For each time step:
        For each remaining project in a random order:
            Add project at latest feasible time step, if possible
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Pref Earliest"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = self.__preferences()

        # order indices by prerequisites
        for candidate_p in indices:
            if self.instance.projects[candidate_p].has_prerequisites():
                pass

        for candidate_p in indices:
            # if project already started, skip (can occur as a result of preloading prerequisites)
            if self.portfolio.scheduled(candidate_p):
                continue
            if self.projects[candidate_p].has_prerequisites():
                # ensure that prerequsites are scheduled first
                pass

            earliest = self.find_earliest(candidate_p)
            # earliest = self.portfolio.find_earliest(self.projects[candidate_p], self.budget,
            #                                         self.capability_stream_budgets, self.initiation_budget,
            #                                         self.ongoing_budget, self.planning_window)
            if earliest > 0:
                self.add_to_portfolio(candidate_p, earliest)

        return self.portfolio

    def __preferences(self):
        pref = np.zeros(self.instance.num_projects)
        for i in range(self.instance.num_projects):
            pref[i] = self.instance.projects[i].total_cost / self.instance.projects[i].total_value

        return pref.argsort()
