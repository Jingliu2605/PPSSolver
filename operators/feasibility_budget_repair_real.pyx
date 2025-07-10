# cython: boundscheck=False, wraparound=False, profile=False
import numpy as np
cimport numpy as np

from problem.portfolio import build_from_array

from problem.project import Project
from problem.project cimport Project

from pymoo.model.repair import Repair


class FeasibilityBudgetRepairReal(Repair):

    def __init__(self, normalize=True, prob_repair=1,  **kwargs):
        self.normalize = normalize
        self.prob_repair = prob_repair


    def _do(self, problem, pop, **kwargs):
        cdef int i, index, proj_index, max_reduction, index_to_remove, t
        cdef double viol_sum, proj_value

        X = pop.get("X")  # get solution vectors
        old = X

        # iterate through all solution vectors
        for i in range(X.shape[0]):
            # x = X[i]
            x = np.round(X[i]).astype(int)
            portfolio, violations = build_from_array(x, problem.instance)
            budget_viols = violations["budget_viols"]
            viol_sum = np.sum(budget_viols)

            # exit if no violations or probability check fails
            if viol_sum <= 0 or np.random.random() > self.prob_repair:
                continue

            #started = np.nonzero(portfolio.result)[0]
            while viol_sum > 0:
                # get list of started projects
                started = np.nonzero(portfolio.result)[0]
                reductions = np.zeros(started.shape[0])
                #max_reduction = 0
                #max_index = 0
                for index in range(started.shape[0]):
                    proj_index = started[index]
                    project = problem.instance.projects[proj_index]
                    reductions[index] = self._calculate_reduction(budget_viols, project,
                                                                  portfolio.start_time(proj_index), problem.instance.budget)
                    if self.normalize:  # normalize using the project value associated with the portfolio
                        start_time = portfolio.start_time(proj_index)
                        # proj_value =
                        #for t in range(project.duration):
                        #    proj_value += project.value[t]

                        reductions[index] /= project.total_value

                # find index of highest reduction
                max_reduction = np.argmax(reductions)

                # extract project index
                index_to_remove = started[max_reduction]

                #TODO: ensure project has no successors, if so, do not remove

                # remove from portfolio, then test for violations again
                portfolio.remove_from_portfolio(index_to_remove, problem.instance.projects[index_to_remove])
                #budget_viols = portfolio.budget_violations(problem.budget)#portfolio.constraint_violations(problem.projects, problem.budget)
                budget_viols = portfolio._generic_budget_violation(problem.instance.budget, portfolio.cost, problem.instance.budget_window)

                viol_sum = np.sum(budget_viols)

            X[i] = portfolio.result
        # new = X/(problem.instance.planning_window+1)
        # print(X)
        new = old * (X>0)
        # print(1)
        pop.set("X", new)
        return pop

    # calculate the reduction in budget constraints associated with removing the specified project
    def _calculate_reduction(self, np.double_t[:] budget_viols, Project project, int start, np.double_t[:] budget):
        cdef int t, abs_index
        cdef double reduction, viol_amount, time_cost

        reduction = 0
        for t in range(project.duration):
            abs_index = start + t - 1
            viol_amount = budget_viols[abs_index]
            time_cost = project.cost_raw.data.as_doubles[t]#project.cost_at_time(t)
            if viol_amount == 0:  # no violation, move to next time step
                continue
            elif time_cost > viol_amount: # if cost is greater than violation amount, increase by violation amount
                reduction += viol_amount
            else: # cost <= violation amount, subtract the cost at this time
                reduction += time_cost

        return reduction

    def __str__(self):
        return f"Feasibility Budget Repair (p={self.prob_repair}, norm={self.normalize})"

    def __repr__(self):
        return self.__str__()
