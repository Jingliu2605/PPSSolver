import numpy as np

from solvers.heuristic_solver import HeuristicSolver


class CyclicSolver(HeuristicSolver):
    """
    Generate a feasible solution using cyclic project selection.

    While not completed:
        For each time step:
            Select feasible project (if available)
            Continue to next time step
        If no projects selected, mark completed
    """

    def __init__(self, instance, seed=1):
        super().__init__(instance, seed)
        self.name = "Cyclic"

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = np.random.permutation(range(self.instance.num_projects))
        done = False

        while not done:
            done = True  # return true if no new project was added in this iteration

            # cycle through time steps and add a project, if possible
            for t in range(1, self.instance.planning_window + 1):
                for index in range(len(indices)):
                    candidate_p = indices[index]
                    if self.feasibility_check(candidate_p, t):
                        self.add_to_portfolio(candidate_p, t)
                        indices = np.delete(indices, index)
                        done = False
                        break  # move to next time step

                indices = np.random.permutation(indices)

        return self.portfolio
