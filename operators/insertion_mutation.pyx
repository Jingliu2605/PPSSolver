import numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_array
from pymoo.model.mutation import Mutation


class InsertionMutation(Mutation):
    """
    For start-time representation
    """

    def __init__(self, scheduling_order=SchedulingOrder.EARLIEST, prob=0.1):
        super().__init__()
        self.prob = prob
        self.scheduling_order = scheduling_order

    def _do(self, problem, X, **kwargs):
        cdef int i, j
        _X = np.copy(X)

        for i in range(X.shape[0]):
            if np.random.random() < self.prob:
                mutant = _X[i]
                portfolio, violations = build_from_array(mutant, problem.instance)
                # probability as 1/non-zero if set to None, otherwise, set to parameter
                #if np.count_nonzero(violations["budget_viols"]) > 0 or violations["prereq_viols"] > 0 or \
                if violations["prereq_viols"] > 0 or violations["exclusion_viols"] > 0 or \
                        np.any(violations["budget_viols"]):
                    continue

                # find indices of the mutant that are 0 (i.e., projects not scheduled)
                zero_indices = np.where(mutant == 0)[0]
                if zero_indices.shape[0] <= 0:  # all projects have been implemented, skip this individual
                    continue

                shuffled = np.random.permutation(zero_indices)

                for j in shuffled:
                    if self.scheduling_order is SchedulingOrder.EARLIEST:
                        t = portfolio.find_earliest(problem.projects[j], problem.budget, problem.planning_window)
                    elif self.scheduling_order is SchedulingOrder.LATEST:
                        t = portfolio.find_latest(problem.projects[j], problem.budget, problem.planning_window)
                    else:
                        t = 0

                    if t > 0:
                        portfolio.add_to_portfolio(j, t, problem.projects[j])
                        mutant[j] = t
        return _X
