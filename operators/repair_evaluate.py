# Jing Liu: considers the capability budget
import numpy as np
# cimport numpy as np

from problem.portfolio import build_from_array

from problem.project import Project
# from problem.project cimport Project

from pymoo.model.repair import Repair
from problem.test_portfolio_with_repair import build_from_array_and_repair


class FeasibilityRepairEvaluate(Repair):

    def __init__(self, normalize=True, prob_repair=1, real_flag=0, **kwargs):
        self.normalize = normalize
        self.prob_repair = prob_repair
        self.real_flag = real_flag
        # real_flag=0, if the algorithm is operated in discrete domains; =1, otherwise

    def _do(self, problem, pop, **kwargs):
        # cdef int i, index, proj_index, max_reduction, index_to_remove, t
        # cdef double viol_sum, proj_value

        X = pop.get("X")  # get solution vectors
        count = X.shape[0]
        fits = np.zeros(count, dtype=np.double)
        viols = [0] * count
        if self.real_flag == 1:
            phenotypes = [None] * count
            hashes = [None] * count

        # iterate through all solution vectors
        for i in range(count):
            if self.real_flag == 1:
                phenotype = np.round(X[i]).astype(int)
                portfolio, violations = build_from_array(phenotype, problem.instance)
            else:
                x = X[i]
                portfolio, violations = build_from_array(x, problem.instance)
                portfolio, violations, new_x = build_from_array_and_repair(x[i], self.instance)

            budget_viols = violations["budget_viols"]
            stream_viols = violations["stream_viols"]
            initiation_viols = violations["initiation_viols"]
            ongoing_viols = violations["ongoing_violations"]
            viol_sum = np.sum(budget_viols)+np.sum(stream_viols)+np.sum(initiation_viols)+np.sum(ongoing_viols)
            # exit if no violations or probability check fails
            if viol_sum <= 0 or np.random.random() > self.prob_repair:
                fits[i] = -portfolio.value
                viols[i] = violations['all_viols']
                if self.real_flag == 1:
                    phenotypes[i] = phenotype
                    hashes[i] = hash(str(phenotype))
                continue

            projects = problem.instance.projects
            started = np.nonzero(portfolio.result)[0]
            # repair if violations exist
            while viol_sum > 0:
                if violations["exclusion_viols"] > 0:
                    for i in range(len(started)):
                        project_index = started[i]
                        exclusions = projects[project_index].exclusion_list
                        exclusion_index = np.intersect1d(exclusions, started)
                        # for index in range(exclusions.shape[0]):


            # budget violations
            # stream violations
            # initiation violations
            # ongoing violations
            #




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

