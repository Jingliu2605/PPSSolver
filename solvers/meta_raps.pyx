import time

import numpy as np
cimport numpy as np

from problem.portfolio import Portfolio


class MetaRaPS:

    def __init__(self, instance, p=0.65, r=0.2, iterations=1000, ordering=None, seed=None):
        self.instance = instance
        self.projects = instance.projects
        self.budget = instance.budget
        self.num_projects = instance.projects.shape[0]
        self.budget_window = instance.budget.shape[0]
        self.planning_window = instance.planning_window
        self.discount_rate = instance.discount_rate
        self.capability_streams = instance.capability_stream_budgets.shape[0]
        self.p = p
        self.r = r
        self.ordering=None
        self.iterations = iterations
        np.random.seed(seed)

    def solve(self):
        start_time = time.perf_counter()
        cdef int i, current_index, upper, index, prerequisite, candidate_p
        cdef np.int_t[:] indices
        cdef np.int_t[:] p_list

        if self.ordering is None:
            indices = self.__preferences().astype(int)
        else:
            indices = self.ordering

        best_portfolio = Portfolio(self.num_projects, self.budget_window, self.planning_window, self.discount_rate,
                                   self.capability_streams)
        index_range = int(self.r * self.num_projects)

        for i in range(self.iterations):
            portfolio = Portfolio(self.num_projects, self.budget_window, self.planning_window, self.discount_rate,
                                   self.capability_streams)
            current_index = 0
            while current_index < self.num_projects:
                #selection_mask = np.ones(self.num_projects, dtype=bool)
                # TODO: remove elements that have been considered via range selection or prerequisites
                if np.random.random() < self.p:
                    candidate_p = indices[current_index]
                    current_index += 1
                else:
                    upper = min(current_index + index_range, self.num_projects)
                    candidate_p = np.random.choice(indices[current_index:upper])
                    #selection_mask[candidate_p] = False
                    # TODO: remove selected project from future consideration

                # if project already started, skip (can occur as a result of preloading prerequisites or ranked selection)
                if portfolio.scheduled(candidate_p):
                    continue

                project = self.projects[candidate_p]
                p_list = project.prerequisite_list
                for index in range(p_list.shape[0]):
                    prerequisite = p_list[index]

                    # TODO: remove projects from future consideration
                    portfolio.add_earliest_feasible(prerequisite, self.projects[prerequisite], self.instance)

                portfolio.add_earliest_feasible(candidate_p, project, self.instance)

                #TODO: add improvement step

            if portfolio.value > best_portfolio.value:
                best_portfolio = portfolio

        end_time = time.perf_counter()
        return best_portfolio, end_time - start_time

    def __preferences(self):
        cdef int i
        pref = np.zeros(self.num_projects)
        projects = self.projects
        #preferences = np.fromiter(p.total_cost / p.total_value for p in self.projects)
        for i in range(self.num_projects):
            project = self.projects[i]
            pref[i] = project.total_cost / project.total_value

        return pref.argsort()
