# cython: boundscheck=False, wraparound=False, profile=False
# Jing Liu
cimport numpy as np
import numpy as np

import problem.portfolio
from pymoo.model.problem import Problem


class PortfolioSelectionProblemWithRepair(Problem):

    def __init__(self, instance, real_flag=0, **kwargs):
        # constraints are 1 per budget year, mutual exclusion, and prerequisites
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=len(instance.budget) + 2, xl=0,
                         xu=instance.planning_window, type_var=int, **kwargs)
        self.instance = instance
        self.real_flag = real_flag
        # real_flag=0, if the algorithm is operated in discrete domains; =1, otherwise

    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef np.ndarray fits = np.zeros(count, dtype=np.double)
        cdef list viols = [0] * count
        cdef list results = [None] * count
        cdef int i
        cdef list phenotypes = [None] * count


        for i in range(count):

            phenotype = np.round(x[i]).astype(int)
            portfolio, violations, new_x, new_phenotype = problem.portfolio.build_from_array_and_repair(phenotype, x[i], self.instance, self.real_flag)
            phenotypes[i] = new_phenotype
            x[i] = new_x
            fits[i] = -portfolio.value # negate as pymoo expects minimization
            viols[i] = violations['all_viols']
            results[i] = portfolio.result

        out["F"] = fits
        out["G"] = np.array(viols)
        out["result"] = np.stack(results)
        out["X"] = x
        out["pheno"] = np.stack(phenotypes)
