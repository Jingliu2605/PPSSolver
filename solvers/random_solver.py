import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class RandomSolver(HeuristicSolver):
    """
    Generate a random, feasible solution. Due to the generation scheme, this heavily front-loads projects

    For each time step:
        For each project in a random order:
            Add project, if feasible
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Random"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = np.random.permutation(range(self.instance.num_projects))
        remaining = []

        for t in range(1, self.instance.planning_window + 1):
            indices = np.random.permutation(indices)
            for candidate_p in indices:
                if self.feasibility_check(candidate_p, t):
                    self.add_to_portfolio(candidate_p, t)
                else:
                    remaining.append(candidate_p)

            # done with this time step, move to next by replacing list
            indices = np.random.permutation(remaining)
            remaining.clear()

        return self.portfolio
