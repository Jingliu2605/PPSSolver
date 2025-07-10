# cython: boundscheck=False, wraparound=False, profile=False

import numpy as np

import problem.portfolio
from problem.portfolio_selection_instance import PortfolioSelectionInstance
from pymoo.model.problem import Problem
from solvers.pref_earliest_solver import PrefEarliestSolver


class HierarchicalSelectionProblem(Problem):

    def __init__(self, instance, **kwargs):
        # constraints are 1 per budget year, mutual exclusion, and prerequisites
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=0, xl=0,
                         xu=1, type_var=bool, **kwargs)
        self.instance = instance

    def _evaluate(self, x, out, *args, **kwargs):
        #cdef int count = x.shape[0]
        count = x.shape[0]
        #cdef list fits = [0] * count
        fits = np.zeros(count, dtype=np.double)
        #cdef np.ndarray fits = np.zeros(count, dtype=np.double)
        #cdef list viols = [0] * count
        viols = [0] * count
        #cdef int i

        for i in range(count):
            portfolio, violations = problem.portfolio.build_from_array(x[i].astype(int), self.instance)

            # if this selection violates constraints, assign violation sum as fitness and do not run heuristic
            viol_sum = violations["budget_viols"].sum() + violations["prereq_viols"] + violations["exclusion_viols"]
            if viol_sum > 0:
                fits[i] = viol_sum
                continue

            # get list of unselected projects, converted to a numpy array
            unselected = np.array(self.instance.projects)[np.nonzero(portfolio.result == 0)]
            # calculate the remaining budget after this portfolio is selected
            new_budget = self.instance.budget - portfolio.cost
            # use heuristic to generate the fitness associated with the future years
            heuristic = PrefEarliestSolver(unselected, new_budget, self.instance.planning_window - 1)
            future_portfolio = heuristic.solve()

            # add heuristic value plus portfolio value to get estimate of fitness
            # negate as pymoo expects minimization
            fits[i] = -(portfolio.value + future_portfolio.value)
            # TODO: can this be simplified?
            #viols[i] = np.append(violations["budget_viols"],
            #                     [violations["prereq_viols"], violations["exclusion_viols"]])
            #viols[i] = np.stack(np.append(np.stack(violations["budget_viols"]),
            #                              [violations["prereq_viols"], violations["exclusion_viols"]]))
        #    costs[i] = np.stack(result["cost"])

        out["F"] = fits  #-np.stack(fits)
        #out["G"] = np.array(viols)
        # out["Cost"] = np.array(costs)  # TODO: how to get the costs assigned to the individuals?
