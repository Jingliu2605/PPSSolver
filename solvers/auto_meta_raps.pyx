import itertools
import time

import numpy as np
cimport numpy as np

from problem.portfolio import Portfolio
from problem.project import Project
from problem.project cimport Project


class AutoMetaRaPS:

    # cdef Project[:] projects
    # cdef const np.double_t[:] budget
    # cdef int num_projects, budget_window, planning_window, tune_repeats, num_exploit, exploit_repeats
    # cdef np.int_t[:] indices

    def __init__(self, instance, int tune_repeats=10, int num_exploit=10, int exploit_repeats=100, ordering=None,
                 seed=None):
        """
        Actual number of evaluations is tune_repeats * 190 + (num_exploit * exploit_repeats)
        :param projects:
        :param budget:
        :param planning_window:
        :param tune_repeats:
        :param num_exploit:
        :param exploit_repeats:
        :param seed:
        """
        self.instance = instance
        self.projects = instance.projects
        self.budget = instance.budget
        self.num_projects = instance.projects.shape[0]
        self.budget_window = instance.budget.shape[0]
        self.planning_window = instance.planning_window
        self.tune_repeats = tune_repeats
        self.num_exploit = num_exploit
        self.exploit_repeats = exploit_repeats
        self.discount_rate = instance.discount_rate
        self.capability_streams = instance.capability_stream_budgets.shape[0]
        self.ordering = ordering
        np.random.seed(seed)

    def solve(self):
        cdef int index, i
        start_time = time.perf_counter()
        best_portfolio = Portfolio(self.num_projects, self.budget_window, self.planning_window, self.discount_rate, \
                                   self.capability_streams)
        if self.ordering is None:
            self.indices = self.__preferences().astype(int)
        else:
            self.indices = self.ordering

        # generate values between [0.05, 1.00]
        p_values = np.arange(0.05, 1.00, 0.05)
        r_values = np.arange(0.05, 0.35, 0.05)

        # get an array of all combinations of p_values and r_values
        mesh = np.array(np.meshgrid(p_values, r_values))
        parameters = mesh.T.reshape(-1, 2)

        # holds the average portfolio value for each parameter configuration
        avg_performance = np.zeros(parameters.shape[0])
        best_performance = np.zeros(parameters.shape[0])
        # temporary storage of the portfolio values from each repeat of parameter configurations
        temp_performance = np.zeros(self.tune_repeats)

        index = 0
        # loop through each parameter configuration, repeating tune_repeats times, and gather performance data
        for (p, r) in parameters:
            for i in range(self.tune_repeats):
                portfolio = self._generate_next(p, r)
                temp_performance[i] = portfolio.value

                if portfolio.value > best_portfolio.value:
                    best_portfolio = portfolio

            avg_performance[index] = temp_performance.mean()
            best_performance[index] = temp_performance.max()
            # print(f"({p},{r}): {avg_performance[index]:0.3f}, {best_performance[index]}")
            index += 1

        # take half to exploit from best average, the other half from best maximum
        num_avg_exploit = self.num_exploit // 2
        num_best_explot = self.num_exploit - num_avg_exploit
        # use argpartition to get the indices of the best num_exploit parameter, then lookup from parameters
        best_avg_indices = np.argpartition(avg_performance, -num_avg_exploit)[-num_avg_exploit:]
        best_max_indices = np.argpartition(best_performance, -num_best_explot)[-num_best_explot:]
        best_params = parameters[np.append(best_avg_indices, best_max_indices)]

        # repeat generation with best parameters
        for (p, r) in best_params:
            for i in range(self.exploit_repeats):
                portfolio = self._generate_next(p, r)

                if portfolio.value > best_portfolio.value:
                    best_portfolio = portfolio

        end_time = time.perf_counter()
        return best_portfolio, end_time - start_time

    def _generate_next(self, double p, double r):
        cdef int current_index, index_range, upper, candidate_p
        portfolio = Portfolio(self.num_projects, self.budget_window, self.planning_window, self.discount_rate,
                              self.capability_streams)
        available = np.ones(self.num_projects, dtype=bool)
        index_range = int(r * self.num_projects)
        current_index = 0
        while current_index < self.num_projects:
            # TODO: remove elements that have been considered via range selection or prerequisites
            if np.random.random() < p:
                candidate_p = self.indices[current_index]
                current_index += 1
            else:
                upper = min(current_index + index_range, self.num_projects)
                candidate_p = np.random.choice(self.indices[current_index:upper])

            if not available[candidate_p]:
                continue

            # skip projects that have already been considered
            available[candidate_p] = False
            # if project already started, skip (can occur as a result of preloading prerequisites or ranked selection)
            if portfolio.scheduled(candidate_p):
                continue

            if self.projects[candidate_p].prerequisite_list.shape[0] > 0:
                # try to schedule prerequisites first
                # TODO: what if prerequisites also have prerequisites?
                for prerequisite in self.projects[candidate_p].prerequisite_list:
                    if not available[prerequisite]:
                        continue
                    portfolio.add_earliest_feasible(prerequisite, self.projects[prerequisite], self.instance)
                    available[prerequisite] = False

            portfolio.add_earliest_feasible(candidate_p, self.projects[candidate_p], self.instance)

        return portfolio

    def __preferences(self):
        cdef int i
        pref =  np.fromiter((p.total_value / p.total_cost for p in self.projects), np.double, count=self.num_projects)

        return pref.argsort()[::-1]
