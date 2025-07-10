import numpy as np

from problem.enums import Weighting, SchedulingOrder
from solvers.heuristic_solver import HeuristicSolver
from util import roulette_wheel_select


class RouletteSolver(HeuristicSolver):
    """
    Generate a semi-greedy feasible solution by selecting projects according to their value-to-cost ratio, using
    roulette wheel selection, then scheduling as early as feasible.
    """

    def __init__(self, instance, seed=1, weighting=Weighting.VALUE,
                 ordering=SchedulingOrder.EARLIEST, alpha=2):
        super().__init__(instance, seed)
        self.name = f"Roulette {ordering.value} ({weighting.value})"
        self.alpha = alpha
        self.weighting = weighting
        self.scheduling_order = ordering

    def solve(self):
        """
        Generate a random solution
        :return: list of integers corresponding to the start times, or 0 if not implemented
        """

        indices = np.arange(self.instance.num_projects)
        if self.weighting is Weighting.VC_RATIO:
            weight = self.__value_cost_ratio()
        elif self.weighting is Weighting.VALUE:
            weight = self.__value()
        elif self.weighting is Weighting.COST:
            weight = self.__cost()

        # boolean mask to indicate projects that are available for selection
        available = np.ones(self.instance.num_projects, bool)

        # keep track of those that need prerequisites to be scheduled - add to available when prerequisites scheduled
        prereq_filter = np.zeros(self.instance.num_projects, bool)
        for i in range(self.instance.num_projects):
            if self.instance.projects[i].prerequisite_list.shape[0] == 0:
                prereq_filter[i] = True

        # consider only those available that have prerequisites met
        selections = np.bitwise_and(available, prereq_filter)
        while selections.any():
            # select a project index, then remove from available
            index = roulette_wheel_select(indices[selections], weight[selections], self.alpha)
            available[index] = False

            # find feasible time based on scheduling order
            # NOTE: earliest scheduling is always used if a project has successors
            if self.scheduling_order is SchedulingOrder.EARLIEST or self.instance.projects[index].successor_list.shape[0] > 0:
                t = self.find_earliest(index)
            else:
                t = self.find_latest(index)
            if t > 0:
                self.add_to_portfolio(index, t)

                # remove any projects on the exclusion list from available list
                for e in self.instance.projects[index].exclusion_list:
                    available[e] = False

                # TODO: assumes that only prerequisite exists
                # make successor projects pass the prerequisite check, such that they can become available
                for s in self.instance.projects[index].successor_list:
                    prereq_filter[s] = True

            # regenerate the list of potential projects
            selections = np.bitwise_and(available, prereq_filter)

        return self.portfolio

    def __value_cost_ratio(self):
        return np.fromiter((np.sum(p.value) / p.total_cost for p in self.instance.projects), np.double, count=self.instance.num_projects)

    def __value(self):
        return np.fromiter((np.sum(p.value) for p in self.instance.projects), np.double, count=self.instance.num_projects)

    def __cost(self):
        return np.fromiter((p.total_cost for p in self.instance.projects), np.double, count=self.instance.num_projects)
