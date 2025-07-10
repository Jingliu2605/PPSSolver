# cython: boundscheck=False, wraparound=False, profile=False

cimport numpy as np
import numpy as np

from problem.portfolio import build_from_array
from pymoo.model.problem import Problem


class PortfolioRealProblem(Problem):

    def __init__(self, instance, **kwargs):
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=len(instance.budget) + 2, xl=0, xu=instance.planning_window, **kwargs)
        self.instance = instance

    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef np.ndarray fits = np.zeros(count, dtype=np.double)
        cdef list phenotypes = [None] * count
        cdef list hashes = [None] * count
        cdef list viols = [0] * count
        cdef int i
        cdef np.int_t[:] phenotype

        for i in range(count):
            #convert to schedule by moving into buckets
            # phenotype = np.trunc(x[i] * (self.instance.planning_window + 1)).astype(int)
            # phenotype = np.round(x[i] * (self.instance.planning_window + 1)).astype(int)
            phenotype = np.round(x[i]).astype(int)
            portfolio, violations = build_from_array(phenotype, self.instance)

            fits[i] = -portfolio.value
            viols[i] = violations['all_viols']
            phenotypes[i] = phenotype
            hashes[i] = hash(str(phenotype))
        #    costs[i] = np.stack(result["cost"])

        out["F"] = fits
        out["G"] = np.array(viols)
        out["pheno"] = np.stack(phenotypes)
        out["hash"] = np.stack(hashes)
    # out["Cost"] = np.array(costs)  # TODO: how to get the costs assigned to the individuals?


#TODO: move this to its own file eventually
from pymoo.model.duplicate import ElementwiseDuplicateElimination

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]

