import scipy.stats as dists
import numpy as np

from problem.enums import ValueFunction, SpreadDistribution


class InstanceParameters:

    def __init__(self, num_projects, planning_window, base_budget, budget_increase,
                 capability_stream_proportions=np.array([1]), initiation_max_proportion=0.25,
                 ongoing_max_proportion=0.75, prerequisite_tuples=[(2, 0.1)], exclusion_tuples=[(2, 0.05), (3, 0.45)],
                 initiation_range=(20, 50), completion_constraint_chance=0,
                 completion_window_size_distribution=dists.randint(1, 10),
                 completion_window_offset=dists.randint(1, 6), value_func=ValueFunction.COST_DUR,
                 cost_distribution=SpreadDistribution.WEIBULL, value_distribution=SpreadDistribution.WEIBULL,
                 discount_rate=0.0, **kwargs):

        self.num_projects = num_projects
        self.planning_window = planning_window
        self.base_budget = base_budget
        self.budget_increase = budget_increase
        self.capability_stream_proportions = capability_stream_proportions

        self.initiation_range = initiation_range
        self.initiation_max_proportion = initiation_max_proportion
        self.ongoing_max_proportion = ongoing_max_proportion

        self.value_func = value_func
        self.cost_distribution = cost_distribution
        self.value_distribution = value_distribution
        self.prerequisite_tuples = prerequisite_tuples
        self.exclusion_tuples = exclusion_tuples
        self.completion_constraint_chance = completion_constraint_chance
        self.completion_window_size_distribution = completion_window_size_distribution
        self.completion_window_offset = completion_window_offset
        self.discount_rate = discount_rate
        self.kwargs = kwargs
