# cython: boundscheck=False, wraparound=False, profile=False

import numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_permutation
from pymoo.model.problem import Problem


class PortfolioPermutationProblem(Problem):
    """
    No constraints as projects are implemented at their earliest feasible time, hence no invalid solutions can be
    constructed.
    """

    def __init__(self, instance, scheduling_order=SchedulingOrder.LATEST, **kwargs):
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=0, xl=0,
                         xu=len(instance.projects), type_var=int, **kwargs)
        self.instance = instance
        self.scheduling_order = scheduling_order

    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef list fits = [0] * count
        cdef list results = [None] * count
        cdef int i
        for i in range(count):
            portfolio = build_from_permutation(x[i], self.instance, self.scheduling_order)
            fits[i] = portfolio.value
            results[i] = portfolio.result

        out["F"] = -np.stack(fits)
        out["result"] = np.stack(results)
