# cython: boundscheck=False, wraparound=False, profile=False

cimport numpy as np
import numpy as np

import problem.portfolio
from pymoo.model.problem import Problem


class PortfolioSelectionProblem(Problem):

    def __init__(self, instance, **kwargs):
        # constraints are 1 per budget year, mutual exclusion, and prerequisites
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=len(instance.budget) + 2, xl=0,
                         xu=instance.planning_window, type_var=int, **kwargs)
        self.instance = instance


    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef np.ndarray fits = np.zeros(count, dtype=np.double)
        cdef list viols = [0] * count
        cdef int i

        for i in range(count):
            portfolio, violations = problem.portfolio.build_from_array(x[i], self.instance)
            #negate as pymoo expects minimization
            fits[i] = -portfolio.value
            # TODO: can this be simplified?

            viols[i] = violations['all_viols']#np.concatenate((violations["budget_viols"], violations["stream_viols"],
                                        # violations["initiation_viols"], [violations["prereq_viols"],
                                                                        #violations["exclusion_viols"]]))

        out["F"] = fits
        out["G"] = np.array(viols)
