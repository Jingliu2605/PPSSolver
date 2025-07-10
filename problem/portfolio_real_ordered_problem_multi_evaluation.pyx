# cython: boundscheck=False, wraparound=False, profile=True
# Jing Liu
cimport numpy as np
import numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_permutation, schedule_earliest_stochastic_limited
from pymoo.model.problem import Problem


class PortfolioRealOrderedProblemMultiEvaluation(Problem):
    """
    No constraints as projects are implemented at their earliest feasible time, hence no invalid solutions can be
    constructed.
    """

    def __init__(self, instance, scheduling_order=SchedulingOrder.LATEST, feas_range=4, p_random=0.01, n_tries=50, **kwargs):
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=0, xl=0, xu=1, **kwargs)

        self.instance = instance
        self.scheduling_order = scheduling_order
        self.feas_range = feas_range
        self.p_random = p_random
        self.n_tries = n_tries

    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef list fits = [0] * count
        cdef list phenotypes = [None] * count
        cdef list results = [None] * count
        cdef list hashes = [None] * count
        cdef int i
        cdef np.int_t[:] phenotype

        cdef list multi_result = [None] * self.n_tries
        cdef list multi_fits = [0] * self.n_tries

        for i in range(count):
            phenotype = np.argsort(x[i]).astype(int)
            for index in range (self.n_tries-1):
                new_portfolio = schedule_earliest_stochastic_limited(phenotype, self.instance, self.p_random, self.feas_range)
                multi_fits[index] = new_portfolio.value
                multi_result[index] = new_portfolio.result
            new_portfolio = build_from_permutation(phenotype, self.instance, self.scheduling_order)
            multi_fits[index+1] = new_portfolio.value
            multi_result[index+1] = new_portfolio.result
            best_index = np.argmax(multi_fits)
            # portfolio = build_from_permutation(phenotype, self.instance, self.scheduling_order)

            fits[i] = multi_fits[best_index]
            phenotypes[i] = phenotype
            results[i] =  multi_result[best_index]
            hashes[i] = hash(str(phenotype))

        out["F"] = -np.stack(fits)
        out["pheno"] = np.stack(phenotypes)
        out["result"] = np.stack(results)
        out["hash"] = np.stack(hashes)

#TODO: move this to its own file eventually
from pymoo.model.duplicate import ElementwiseDuplicateElimination

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]

