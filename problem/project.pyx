# cython: boundscheck=False, wraparound=False, profile=False

cimport numpy as np
import numpy as np
import scipy.stats as dists

from problem.datagen import mvlnorm_generate_costdur, ramped_cost_per_year, even_cost_per_year, \
    fuzzy_weibull_cost_distribution
from problem.enums import SpreadDistribution
from problem.value_functions import get_value_from_enum
from util import nparray_tostring_helper, roulette_wheel_select

cdef class Project:

    @property
    def cost(self):
        """
        Property for cost. Note: converts from internal array representation to numpy on access - use sparingly.
        :return: Array of yearly costs.
        """
        return np.array(self.cost_raw, dtype=np.double)

    def cost_at_time(self, int t):
        return self.cost_raw.data.as_doubles[t]

    def value_at_time(self, int t):
        return self.value[t]

    def __init__(self, str project_name, np.double_t[:] cost, np.double_t[:] value, int duration, double total_cost,
                 np.ndarray prerequisite_list, np.ndarray successor_list, np.ndarray exclusion_list, tuple completion_window,
                 int capability_stream=0):
        self.project_name = project_name
        self.cost_raw = array.array('d', cost.base)
        self.value = value
        self.duration = duration
        self.prerequisite_list = prerequisite_list
        self.successor_list = successor_list
        self.exclusion_list = exclusion_list
        self.total_cost = total_cost
        self.completion_window = completion_window
        self.total_value = np.sum(value) # TODO: this does not account for time discounting
        self.capability_stream = capability_stream

    def copy(self):
        return Project(self.project_name, np.copy(np.asarray(self.cost_raw)), np.copy(np.asarray(self.value)), self.duration,
                       self.total_cost, np.copy(np.asarray(self.prerequisite_list)), np.copy(np.asarray(self.successor_list)),
                       np.copy(np.asarray(self.exclusion_list)), self.completion_window, self.capability_stream)

    def __str__(self):
        return f"{self.project_name}\tCS:{self.capability_stream}\tC:{nparray_tostring_helper(self.cost)}\tV:{nparray_tostring_helper(self.value)}\tD:{self.duration}\t" \
               f"P:{nparray_tostring_helper(self.prerequisite_list)}\tS:{nparray_tostring_helper(self.successor_list)}\t" \
               f"E:{nparray_tostring_helper(self.exclusion_list)}\tCW:{self.completion_window}"

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        """
        Define how to pickle project, used during parallelization.
        :return:
        """
        state = dict()
        state['project_name'] = self.project_name
        state['_cost'] = self.cost_raw.tobytes()
        state['value'] = np.asarray(self.value)
        state['duration'] = self.duration
        state['prerequisite_list'] = np.asarray(self.prerequisite_list)
        state['successor_list'] = np.asarray(self.successor_list)
        state['exclusion_list'] = np.asarray(self.exclusion_list)
        state['total_cost'] = self.total_cost
        state['completion_window'] = self.completion_window
        state['capability_stream'] = self.capability_stream

        return state

    def __setstate__(self, state):
        """
        Define how to unpickle a project.
        :param state:
        :return:
        """
        self.cost_raw = array.array('d')
        self.cost_raw.frombytes(state['_cost'])
        self.project_name = state['project_name']
        self.value = state['value']
        self.duration = state['duration']
        self.prerequisite_list = state['prerequisite_list']
        self.successor_list = state['successor_list']
        self.exclusion_list = state['exclusion_list']
        self.total_cost = state['total_cost']
        self.completion_window = state['completion_window']
        self.capability_stream = state['capability_stream']


def create_random_projects_from_param(params, seed=1):
    """
    Create a list of random projects using a parameter object to specify configuration.

    :param params: An InstanceParameters object specifying the project generation configuration.
    :param seed: The random seed
    :return: A numpy array of Project objects
    """

    return create_random_projects(params.num_projects, params.value_func, params.cost_distribution, params.value_distribution,
                                  params.prerequisite_tuples, params.exclusion_tuples, params.completion_constraint_chance,
                                  params.completion_window_size_distribution, params.completion_window_offset,
                                  params.capability_stream_proportions, seed, **params.kwargs)

def create_random_projects(num_projects, value_function, cost_distribution = SpreadDistribution.WEIBULL,
                           value_distribution=SpreadDistribution.WEIBULL,
                           prerequisite_tuples=None, exclusion_tuples=None,
                           completion_constraint_chance = 0,completion_window_size_distribution = dists.randint(1, 10),
                           completion_window_offset = dists.randint(1, 6), capability_stream_proportions=[1],
                           seed=1, **kwargs):
    """
    Create a list of random projects.

    :param num_projects: Number of projects
    :param value_function: Function to generate values
    :param cost_distribution: Enum specifying the cost distribution
    :param prerequisite_tuples: Tuples representing the number and chance of a project having prerequisite(s).
    The tuple (g, p) denotes that g prerequisites will be generated for proportion p of projects.
    :param exclusion_tuples: Tuples representing the number and chance of a project having mutual exclusion(s).
    The tuple (g, p) denotes that a mutual exclusion group of size g will be generated for proportion p of projects.
    :param completion_constraint_chance: Chance of a project having a constraint completion time
    :param completion_window_size_distribution: Distribution for the length of the completion window.
    :param completion_window_offset: Distribution for the offset used to calculate the completion window.
        :param capability_stream_proportions:
    :param seed: The random seed
    :param kwargs: Other keyword arguments
    :return: A numpy array of Project objects
    """
    np.random.seed(seed)  # scipy uses numpy to generate random
    projects = np.empty(num_projects, dtype=object)

    cost_dur = mvlnorm_generate_costdur(num_projects)
    max_dur = 0

    capability_streams = np.arange(capability_stream_proportions.shape[0])

    for i in range(num_projects):
        duration = cost_dur[i, 0]
        total_cost = cost_dur[i, 1]

        if cost_distribution is SpreadDistribution.RAMPED:
            cost = ramped_cost_per_year(total_cost, duration)
        elif cost_distribution is SpreadDistribution.EVEN:
            cost = even_cost_per_year(total_cost, duration)
        elif cost_distribution is SpreadDistribution.WEIBULL:
            cost = fuzzy_weibull_cost_distribution(total_cost, duration)

        # distribute cost evenly over each year as initial value over time implementation
        total_value = get_value_from_enum(value_function, total_cost, duration, **kwargs)
        if value_distribution is SpreadDistribution.RAMPED:
            value = ramped_cost_per_year(total_value, duration)
        elif value_distribution is SpreadDistribution.EVEN:
            value = even_cost_per_year(total_value, duration)
        elif value_distribution is SpreadDistribution.WEIBULL:
            value = fuzzy_weibull_cost_distribution(total_value, duration)

        #yearly_value /= duration
        #value = np.full(duration, round(yearly_value), dtype=np.double)
        # round to nearest integer
        value = value.round()

        completion_window = _generate_completion_window(duration, completion_constraint_chance,
                                                        completion_window_size_distribution, completion_window_offset)
        exclusions = np.empty(0, dtype=int)
        prerequisites = np.empty(0, dtype=int)
        successors = np.empty(0, dtype=int)

        capability_stream = roulette_wheel_select(capability_streams, capability_stream_proportions)

        # TODO: generate start and end windows -> must ensure that prerequisites are still valid
        projects[i] = Project(f"Project {i + 1}", cost, value, duration, total_cost,
                              prerequisites, successors, exclusions, completion_window, capability_stream)

    if exclusion_tuples is not None:
        _generate_exclusions_post(projects, exclusion_tuples)

    if prerequisite_tuples is not None:
        _generate_prerequisites_post(projects, prerequisite_tuples)

    return projects

def _generate_prerequisites_post(projects, group_sizes):
    num_projects = projects.shape[0]
    indices = np.arange(num_projects)

    for size, prop in group_sizes:
        num_groups = int(prop * num_projects / size)

        # TODO: filter projects that were already included in exclusions
        groups = np.random.choice(indices, size=(num_groups, size), replace=False)

        # remove projects from being selected in future rounds
        indices = np.setdiff1d(indices, groups)

        for group in groups:
            #sort group, such that the
            sorted_group = np.sort(group)
            prereq = sorted_group[0] # first element is the prerequisite
            for i in range(1, size): # remaining elements are successors
                successor = sorted_group[i]
                projects[prereq].successor_list = np.append(projects[prereq].successor_list, successor)
                projects[successor].prerequisite_list = np.append(projects[successor].prerequisite_list, prereq)

def _generate_exclusions_post(projects, group_sizes):
    num_projects = projects.shape[0]
    indices = np.arange(num_projects)

    for size, prop in group_sizes:
        num_groups = int(prop * num_projects / size)

        # TODO: filter projects that were already included in exclusions
        groups = np.random.choice(indices, size=(num_groups, size), replace=False)

        # remove projects from being selected in future rounds
        indices = np.setdiff1d(indices, groups)

        for group in groups:
            for i in range(size):
                ind1 = group[i]
                # add all projects (other than i) to exclusion list
                for j in range(size):
                    if i == j:
                        continue
                    ind2 = group[j]
                    projects[ind1].exclusion_list = np.append(projects[ind1].exclusion_list, ind2)


def _generate_completion_window(duration, completion_constraint_chance, completion_window_size_distribution,
                                completion_window_offset):
    completion_window = None

    #TODO: must consider prerequisites when generating completion windows to prevent projects that can not be scheduled
    # completion time must be minimum of prerequisite.window_start + duration

    if np.random.random() < completion_constraint_chance:
        window_size = completion_window_size_distribution.rvs()
        offset = completion_window_offset.rvs()
        window_start = duration + offset
        window_end = window_start + window_size
        completion_window = (window_start, window_end)

    return completion_window

