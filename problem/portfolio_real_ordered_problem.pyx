# cython: boundscheck=False, wraparound=False, profile=True

cimport numpy as np
import numpy as np

from problem.enums import SchedulingOrder
from problem.portfolio import build_from_permutation
from pymoo.model.problem import Problem


class PortfolioRealOrderedProblem(Problem):
    """
    No constraints as projects are implemented at their earliest feasible time, hence no invalid solutions can be
    constructed.
    """

    def __init__(self, instance, scheduling_order=SchedulingOrder.LATEST, pause_event=None, stop_event=None, **kwargs):
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=0, xl=0, xu=1, **kwargs)

        self.instance = instance
        self.scheduling_order = scheduling_order
        self.pause_event = pause_event
        self.stop_event = stop_event

    def _evaluate(self, x, out, *args, **kwargs):
        cdef int count = x.shape[0]
        cdef list fits = [0] * count
        cdef list phenotypes = [None] * count
        cdef list results = [None] * count
        cdef list hashes = [None] * count
        cdef int i
        cdef np.int_t[:] phenotype

        for i in range(count):
            phenotype = np.argsort(x[i]).astype(int)
            portfolio = build_from_permutation(phenotype, self.instance, self.scheduling_order)

            fits[i] = portfolio.value
            phenotypes[i] = phenotype
            results[i] = portfolio.result
            hashes[i] = hash(str(phenotype))

            if self.stop_event and self.stop_event.is_set():
                print("Solver stopped.")
                return None  # Indicate that the solver was stopped

            if self.pause_event:
                self.pause_event.wait()  # Wait if paused

        out["F"] = -np.stack(fits)
        out["pheno"] = np.stack(phenotypes)
        out["result"] = np.stack(results)
        out["hash"] = np.stack(hashes)


#TODO: move this to its own file eventually
from pymoo.model.duplicate import ElementwiseDuplicateElimination

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]

