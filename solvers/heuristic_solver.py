import time
from abc import abstractmethod

import numpy as np

from problem.portfolio import Portfolio
from util import combine_dicts


class HeuristicSolver:
    """
    Attributes
    ----------
    projects : array of projects
        a list of projects
    budget : array of numeric
        the budget for each time step
    num_projects : int
        the number of projects to be considered
    planning_window : int
        the length of time to consider planning
    budget_window : int
        the length of time to consider the budget
    """

    def __init__(self, instance, seed=1):
        self.instance = instance
        np.random.seed(seed)
        self._initialize()

    def _initialize(self):
        self.portfolio = Portfolio(self.instance.num_projects, self.instance.budget_window,
                                   self.instance.planning_window, self.instance.discount_rate,
                                   self.instance.capability_stream_budgets.shape[0])

    @abstractmethod
    def solve(self):
        pass

    def repeat(self, repetitions, display=True):
        """Run the heuristic solver a number of times and aggregate the results"""
        value = 0
        values = np.zeros(repetitions)
        sum_counts = dict()
        solutions = np.ndarray((repetitions, self.instance.num_projects), dtype=int)
        start_time = time.perf_counter()
        for i in range(repetitions):
            self._initialize()
            solution = self.solve()
            solutions[i] = np.array(solution.result, dtype=int)
            counts = self.start_count()
            sum_counts = combine_dicts(counts, sum_counts)
            value += solution.value
            values[i] = solution.value

        end_time = time.perf_counter()
        for key in sum_counts:
            sum_counts[key] /= repetitions

        if display:
            # print(f"{np.mean(values):0.3f} ({np.std(values):0.3f})", end=" & ")
            print(f"\n--------------{self.name}--------------")
            print(f"Average: {value / repetitions:.3f}")
            print(f"Std: {np.std(values):.3f}")
            print(f"Min: {np.min(values):.3f}")
            print(f"Max: {np.max(values):.3f}")
            print(f"Avg. Time: {(end_time - start_time) / repetitions:.3E}s")
            print(f"Tot. Time: {end_time - start_time:.3E}s")

        return solutions
        # print(sum_counts)

    def add_to_portfolio(self, candidate_p, t):
        """Add a project to the portfolio, update completed time, value, and cost."""
        self.portfolio.add_to_portfolio(candidate_p, t, self.instance.projects[candidate_p])

    def feasibility_check(self, candidate_p, t):
        return self.portfolio.feasibility_check(self.instance.projects[candidate_p], t, self.instance)

    def find_earliest(self, candidate_p):
        """
        Find the earliest time-step where a project can be scheduled

        :return: the earliest time step that the project can be scheduled or -1 if project can not be scheduled
        """
        for t in range(1, self.instance.planning_window + 1):
            if self.portfolio.feasibility_check(self.instance.projects[candidate_p], t, self.instance):
                return t

        return -1  # this indicates that a project can not be scheduled without violating constraints

    def find_latest(self, candidate_p):
        """
        Find the latest time-step where a project can be scheduled

        :return: the latest time step that the project can be scheduled or -1 if project can not be scheduled
        """
        for t in range(self.instance.planning_window, 0, -1):
            if self.portfolio.feasibility_check(self.instance.projects[candidate_p], t, self.instance):
                return t

        return -1  # this indicates that a project can not be scheduled without violating constraints

    def start_count(self):
        """
        Count the number of projects that start at each time step.

        :return: A dictionary with a count of the number of projects that start at each time step.
        """
        result = {}
        for i in range(self.instance.planning_window + 1):
            result[i] = 0

        for i in self.portfolio.result:
            result[i] += 1

        return result
