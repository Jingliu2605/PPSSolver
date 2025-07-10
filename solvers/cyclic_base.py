from abc import abstractmethod

import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class CyclicBaseSolver(HeuristicSolver):
    """
    Generate a feasible solution using cyclic project selection.

    While not completed:
        For each time step:
            Select feasible project by preference (if available)
            Continue to next time step
        If no projects selected, mark completed
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = self._preferences()

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

    @abstractmethod
    def _preferences(self):
        pass
