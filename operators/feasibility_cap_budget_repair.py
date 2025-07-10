# Jing Liu: considers the capability budget
import numpy as np
# cimport numpy as np

from problem.portfolio import build_from_array

from problem.project import Project
# from problem.project cimport Project

from pymoo.model.repair import Repair

#Todo: not finished yet
class FeasibilityCapBudgetRepair(Repair):

    def __init__(self, normalize=True, prob_repair=1, real_flag=0, **kwargs):
        self.normalize = normalize
        self.prob_repair = prob_repair
        self.real_flag = real_flag
        # real_flag=0, if the algorithm is operated in discrete domains; =1, otherwise

    def _do(self, problem, pop, **kwargs):
        # cdef int i, index, proj_index, max_reduction, index_to_remove, t
        # cdef double viol_sum, proj_value

        X = pop.get("X")  # get solution vectors

        # iterate through all solution vectors
        for i in range(X.shape[0]):
            # Todo: use pheno or x ?
            if self.real_flag == 1:
                x = np.round(X[i]).astype(int)
            else:
                x = X[i]
            portfolio, violations = build_from_array(x, problem.instance)
            budget_viols = violations["budget_viols"]
            stream_viols = violations["stream_viols"]
            initiation_viols = violations["initiation_viols"]
            ongoing_viols = violations["ongoing_violations"]
            viol_sum = np.sum(budget_viols)+np.sum(stream_viols)+np.sum(initiation_viols)+np.sum(ongoing_viols)
            # exit if no violations or probability check fails
            if viol_sum <= 0 or np.random.random() > self.prob_repair:
                continue

            while viol_sum > 0:
                # get list of started projects
                started = np.nonzero(portfolio.result)[0]
                reductions = np.zeros(started.shape[0])
                #max_reduction = 0
                #max_index = 0
                for index in range(started.shape[0]):
                    proj_index = started[index]
                    project = problem.instance.projects[proj_index]
                    reductions[index] = self._calculate_all_reduction(budget_viols, stream_viols, initiation_viols, ongoing_viols, project,
                                                                  portfolio.start_time(proj_index), problem.instance.budget)
                    if self.normalize:  # normalize using the project value associated with the portfolio
                        start_time = portfolio.start_time(proj_index)
                        # proj_value =
                        #for t in range(project.duration):
                        #    proj_value += project.value[t]
                        # project.total_value
                        reductions[index] /= sum(project.value)

                # find index of highest reduction
                max_reduction = np.argmax(reductions)

                # extract project index
                index_to_remove = started[max_reduction]

                #TODO: ensure project has no successors, if so, do not remove

                # remove from portfolio, then test for violations again
                portfolio.remove_from_portfolio(index_to_remove, problem.instance.projects[index_to_remove])
                violations=portfolio.constraint_violations(problem.instance)
                budget_viols = violations["budget_viols"]
                stream_viols = violations["stream_viols"]
                initiation_viols = violations["initiation_viols"]
                ongoing_viols = violations["ongoing_violations"]

                # budget_viols = portfolio._generic_budget_violation(problem.instance.budget, portfolio.cost, problem.instance.budget_window)
                # stream_viols = portfolio._generic_budget_violation(problem.instance.capability_stream_budgets, portfolio.capability_stream_costs,
                #                                           len(problem.instance.capability_stream_budgets))
                # initiation_viols = portfolio._generic_budget_violation(problem.instance.initiation_budget, portfolio.start_costs,
                #                                                   problem.instance.planning_window)
                # ongoing_viols = portfolio._generic_budget_violation(problem.instance.ongoing_budget, portfolio.ongoing_costs,
                #                                                problem.instance.budget_window)

                viol_sum = np.sum(budget_viols)+np.sum(stream_viols)+np.sum(initiation_viols)+np.sum(ongoing_viols)

            X[i] = portfolio.result
        pop.set("X", X)
        # print(1)
        return pop

    # calculate the reduction in budget constraints associated with removing the specified project
    def _calculate_reduction(self,  budget_viols, project, start, budget):
        # cdef int t, abs_index
        # cdef double reduction, viol_amount, time_cost

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

    # calculate the reduction in budget constraints associated with removing the specified project
    def _calculate_all_reduction(self, budget_viols, stream_viols, initiation_viols, ongoing_viols, project, start, budget):
        # cdef int t, abs_index
        # cdef double reduction, viol_amount, time_cost

        reduction = 0
        # calculate the stream_viols reduction
        stream_viol_amount = stream_viols[project.capability_stream]
        reduction = self.__calculate_deduction(stream_viol_amount, project.total_cost, reduction)

        for t in range(project.duration):
            abs_index = start + t - 1

            time_cost = project.cost_raw[t]  # project.cost_at_time(t) .data.as_doubles

            if t == 0:
                # calculate the initiation_viols reduction
                initiation_viol_amount = initiation_viols[start - 1]
                reduction = self.__calculate_deduction(initiation_viol_amount, time_cost, reduction)
            else:
                # calculate the ongoing_viols reduction
                ongoing_viol_amount = ongoing_viols[abs_index]
                reduction = self.__calculate_deduction(ongoing_viol_amount, time_cost, reduction)
            # calculate the budget_viols reduction
            budget_viol_amount = budget_viols[abs_index]
            reduction = self.__calculate_deduction(budget_viol_amount, time_cost, reduction)

        return reduction

    def __calculate_deduction(self, viol_amount, time_cost, single_reduction):
        if viol_amount == 0:  # no violation, move to next time step
            return  single_reduction
        elif time_cost > viol_amount:  # if cost is greater than violation amount, increase by violation amount
            single_reduction += viol_amount
        else:  # cost <= violation amount, subtract the cost at this time
            single_reduction += time_cost

        return single_reduction


    def __str__(self):
        return f"Feasibility Budget Repair (p={self.prob_repair}, norm={self.normalize})"

    def __repr__(self):
        return self.__str__()
