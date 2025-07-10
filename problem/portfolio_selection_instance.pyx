# cython: profile=False
import json
import os
import pickle
import re


import numpy as np

from problem.project import Project, create_random_projects_from_param
from util import nparray_tostring_helper


cdef class PortfolioSelectionInstance:

    def __init__(self, projects, budget, capability_stream_budgets, initiation_budget, ongoing_budget, initiation_range,
                 planning_window, discount_rate, parameters, identifier=1):
        self.projects = projects
        self.budget = budget
        self.capability_stream_budgets = capability_stream_budgets
        self.initiation_budget = initiation_budget
        self.ongoing_budget = ongoing_budget

        self.planning_window = planning_window
        self.identifier = str(identifier)
        self.num_projects = projects.shape[0]
        self.budget_window = budget.shape[0]
        self.discount_rate = discount_rate
        self.parameters = parameters

        self.initiation_range = self._fix_initiation_range(initiation_range)

    def __getstate__(self):
        """
        Define how to pickle project, used during parallelization.
        :return:
        """
        state = dict()
        state['projects'] = np.asarray(self.projects)
        state['budget'] = np.asarray(self.budget)
        state['capability_stream_budgets'] = np.asarray(self.capability_stream_budgets)
        state['initiation_budget'] = np.asarray(self.initiation_budget)
        state['ongoing_budget'] = np.asarray(self.ongoing_budget)
        state['initiation_range'] = self.initiation_range

        state['planning_window'] = self.planning_window
        state['identifier'] = self.identifier
        state['num_projects'] = self.num_projects
        state['budget_window'] = self.budget_window
        state['discount_rate'] = self.discount_rate
        state['parameters'] = self.parameters

        return state

    def __setstate__(self, state):
        """
        Define how to unpickle a project.
        :param state:
        :return:
        """
        self.projects = state['projects']
        self.budget = state['budget']
        self.capability_stream_budgets = state['capability_stream_budgets']
        self.initiation_budget = state['initiation_budget']
        self.ongoing_budget = state['ongoing_budget']
        self.initiation_range = state['initiation_range']

        self.planning_window = state['planning_window']
        self.identifier = state['identifier']
        self.num_projects = state['num_projects']
        self.budget_window = state['budget_window']
        self.discount_rate = state['discount_rate']
        self.parameters = state['parameters']

    def _fix_initiation_range(self, initiation_range):

        # no constraints on initiation range
        if initiation_range is None:
            return -1, self.num_projects + 1

        # set to minimum possible value if not given
        if initiation_range[0] is None:
            initiation_min = -1
        else:
            initiation_min = initiation_range[0]

        #set to maximum possible value if not given
        if initiation_range[1] is None:
            initiation_max = self.num_projects + 1
        else:
            initiation_max = initiation_range[1]

        return initiation_min, initiation_max

    def write_to_pickle(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def to_json(self, json_indent=2):
        """
        Convert the project instance to JSON.

        :return: A JSON string
        """
        return json.dumps(self.to_json_dict(), indent=json_indent)

    # Note: not correct
    def to_json_dict(self):
        """
        Convert the PortfolioSelectionInstance object to a dictionary that is JSON serializable.

        :return: A dictionary suitable for converting to JSON.
        """
        output = dict()
        output['problem_name'] = self.identifier
        output['num_projects'] = self.num_projects
        output['periods'] = self.planning_window
        output['budget'] = self.budget
        output['initiation_budget'] = self.initiation_budget
        output['maintenance_budget'] = self.ongoing_budget
        output['capability_stream_budgets'] = self.capability_stream_budgets
        output['projects'] = [p.to_json_dict() for p in self.projects]

        output['discount_rate']=self.discount_rate
        output['budget_window'] = self.budget_window
        output['parameters'] = self.parameters
        output['initiation_range'] = self.initiation_range

        return output

    def write_to_file(self, filename):
        # suppress scientific notation and force 3 decimal places
        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:0.3f}'.format},
                            floatmode='unique')
        with open(filename, "w") as f:
            f.write(f"{len(self.projects)}\t{self.planning_window}\n")
            budget_str = nparray_tostring_helper(self.budget)
            f.write(f"{budget_str}\n")
            f.write('\n'.join(str(p) for p in self.projects))

        # reset numpy printing options to default
        np.set_printoptions()

    def identifier_string(self):
        return f"{self.num_projects}_{self.planning_window}_{self.budget[0]}_{self.budget[1] - self.budget[0]}_{self.identifier}"

    def prefilter(self):
        print("Scanning for infeasible projects to remove from instance.")
        cdef int i
        to_keep = np.ones(self.num_projects, dtype=bool)
        for i in range(self.num_projects):
            p = self.projects[i]
            removed = []
            # check if initiation cost is less than maximum available starting budget
            if p.cost[0] > self.initiation_budget[self.planning_window - 1]:
                print(f"'{p.project_name}' (index {i}) has initiation cost > maximum")
                to_keep[i] = False
                removed.append(self.successor_post_filter(p.successor_list))

            # check if ongoing cost is less than maximum ongoing budget
            elif max(p.cost) > max(self.ongoing_budget):
                print(f"'{p.project_name}' (index {i}) has ongoing cost > maximum")
                to_keep[i] = False
                removed.append(self.successor_post_filter(p.successor_list))

            # check if capability stream budget can be met
            elif p.cost[0:self.planning_window].sum() > self.capability_stream_budgets[p.capability_stream]:
                print(f"'{p.project_name}' (index {i}) has cost > capability stream budget")
                to_keep[i] = False
                removed.append(self.successor_post_filter(p.successor_list))

            # check that all prerequisites can be met
            for prereq in p.prerequisite_list:
                prereq_project = self.projects[prereq]
                if prereq_project.duration >= self.planning_window:
                    print(f"'{p.project_name}' (index {i}) has prerequisite with duration of {prereq_project.duration}")
                    to_keep[i] = False
                    removed.append(self.successor_post_filter(p.successor_list))

            for r in removed:
                to_keep[r] = False

        print(f"Removed indices: {np.nonzero(to_keep == False)[0]}")

        # must convert to array as memory view cannot be sliced.
        filtered_projects = np.copy(np.asarray(self.projects)[to_keep])

        for i in range(filtered_projects.shape[0]):
            filtered_projects[i] = filtered_projects[i].copy()
        #filtered_projects = np.fromiter((np.copy(x) for x in np.asarray(self.projects)[to_keep]), dtype=np.object)

        # create map of new indices for projects
        new_id_map = -np.ones(self.num_projects, dtype=int)
        current_index = 0
        for i in range(self.num_projects):
            if to_keep[i]:
                new_id_map[i] = current_index
                current_index += 1

        # loop through filtered projects and update constraint IDs accordingly
        filtered_index = 0
        for i in range(filtered_projects.shape[0]):
            #if not new_id_map[i] == -1:
            p = filtered_projects[i]
            me_list = []
            p_list = []
            s_list = []
            for me in p.exclusion_list:
                # if not removed, add new id to list
                if not new_id_map[me] == -1:
                    me_list.append(new_id_map[me])

            for prereq in p.prerequisite_list:
                if not new_id_map[prereq] == -1:
                    p_list.append(new_id_map[prereq])

            for succ in p.successor_list:
                if not new_id_map[succ] == -1:
                    s_list.append(new_id_map[succ])

            p.exclusion_list = np.asarray(me_list).astype(int)
            p.prerequisite_list = np.asarray(p_list).astype(int)
            p.successor_list = np.asarray(s_list).astype(int)

                #filtered_index += 1

        # update constraint IDs in remaining projects (mutual exclusion, prerequisite, successors)

        new_instance = PortfolioSelectionInstance(filtered_projects, self.budget, self.capability_stream_budgets,
                                                  self.initiation_budget, self.ongoing_budget, self.initiation_range,
                                                  self.planning_window, self.discount_rate, self.parameters,
                                                  f"{self.identifier}_filtered")

        # need to ensure that constraints are updated accordingly..
        # i.e., remove this project from any other mutual exclusion, prerequisite, or successor lists

        print(f"Built new instance with {new_instance.num_projects} projects "
              f"({self.num_projects - new_instance.num_projects} removed).")
        return new_instance

    def successor_post_filter(self, successor_list):
        removed = []
        for succ in successor_list:
            print(f"\tProject {succ} infeasible as successor.")
            removed.append(succ)
        return removed

def instance_from_pickle(filename):
    with open(filename, "rb") as f:
        instance = pickle.load(f)
    return instance


def portfolio_selection_instance_from_file(filename):
    if not os.path.isfile(filename):
        print(f"File '{filename}' does not exist")
        return None

    base_file = os.path.basename(filename)
    ind1 = base_file.rfind("_")
    ind2 = base_file.rfind(".")
    identifier = base_file[ind1 + 1:ind2] if ind1 > 0 else "0"

    # TODO: DeprecationWarning: string or file could not be read to its end due to unmatched data; this will raise a ValueError in the future.
    # noinspection PyBroadException
    try:
        with open(filename, "r") as file:
            line = file.readline()
            header = line.split()
            num_projects = int(header[0])
            planning_window = int(header[1])
            budget_str = file.readline().strip("[]")  # read the budget and remove the parentheses
            budget = np.fromstring(budget_str, sep=" ")
            project_list = np.empty(num_projects, dtype=object)  #[None] * num_projects
            # use regular expression to parse project line into needed info
            pattern = re.compile(
                r"([^\t]*)\t*C:\[(.*)\]\s*V:\[(.*)\]\s*D:(\d*)\s*P:\[(.*)\]\s*S:\[(.*)\]\s*E:\[(.*)\]\s*CW:(.*)\s*")
            #pattern = re.compile(r"([^\t]*)\t*C:\[(.*)\]\s*V:([\d.]*)\s*D:(\d*)\s*P:\[(.*)\]\s*E:\[(.*)\]\s*CW:(.*)")
            r"""
            (...) -> capture group
            . -> Matches any character (except line terminators)
            * -> Match the preceding expression zero or more times (greedy)
            \t -> tab literal
            \s -> any whitespace character (\r, \n, \t, \f, \v)
            \d -> any digit (0-9)
            \[ \] -> square bracket literals

            ([^\t]*) -> Match the project name as 0 or more characters (except tab) -> Group 1
            C:\[(.*)\] -> Match cost as C:[<zero or more characters>] -> Group 2
            V:\[(.*)\] -> Match value as V:[<zero or more characters>] -> Group 3
            D:(\d*) -> Match duration as D:[<zero or more digits>] -> Group 4
            P:\[(.*)\] -> Match prerequisites as P:[<zero or more characters>] -> Group 5
            S:\[(.*)\] -> Match successors as S:[<zero or more characters>] -> Group 6
            E:\[(.*)\] -> Match exclusions as E:[<zero or more characters>] -> Group 7
            CW:(.*) -> Match completion window as CW:<zero or more characters> -> Group 8
            $ -> Matches the end of a string

            Notes:
            -For all array parsing, grouping captures only the contents within the parentheses 
            (for use with numpy fromstring()).
            -Regarding V, the period (.) does not need to be escaped when inside a character class, i.e., []. 
            Hence the '.' in [\d.]* matches only the '.' and not any character.
            """
            for i in range(num_projects):
                project_str = file.readline()
                match = pattern.match(project_str)
                name = match.group(1)
                cost = np.fromstring(match.group(2), sep=" ")
                value = np.fromstring(match.group(3), sep=" ")
                duration = int(match.group(4))
                prerequisites = np.fromstring(match.group(5), dtype=int, sep=" ")
                successors = np.fromstring(match.group(6), dtype=int, sep=" ")
                exclusions = np.fromstring(match.group(7), dtype=int, sep=" ")
                completion_window = eval(match.group(8))

                successors = np.empty(0, dtype=int)
                # TODO: do we need this? should already be included in the list when written...
                #for e in exclusions:
                #    project_list[e].exclusion_list = np.append(project_list[e].exclusion_list, i)

                #for p in prerequisites:
                #    project_list[p].successor_list = np.append(project_list[p].successor_list, i)

                project_list[i] = Project(name, cost, value, duration, cost.sum(), prerequisites, exclusions,
                                          successors,
                                          completion_window)

        # TODO: add discount factor to read_from_file
        return PortfolioSelectionInstance(project_list, budget, planning_window, 0, identifier)
    except Exception as ex:  # TODO: capture more specific exceptions
        print(f"Error parsing file '{filename}' as project list.")

# TODO clean generate_instance to better fit IIP data generation scheme
def generate_instance(param, random_seed, fuzz=True, identifier=1):

    random_project_list = create_random_projects_from_param(param, random_seed)

    max_length = 0
    for p in random_project_list:
        if p.duration > max_length:
            max_length = p.duration

    param.max_proj_length = max_length

    budget_period = param.planning_window + param.max_proj_length
    budget = np.zeros(budget_period)
    # generate fluctuating budget for 3PI （Jing）

    # steps = np.random.normal(0, 4000, 10)  # Normal distribution step sizes
    # series = np.cumsum(steps)
    noise = np.random.uniform(-param.budget_increase, param.budget_increase, budget_period)
    for y in range(budget_period):
        budget[y] = param.base_budget + param.budget_increase*5 + param.budget_increase * np.sin(
            2 * np.pi * (y+1) / 15) + noise[y]/5
    # for y in range(budget_period):
    #     budget[y] = param.base_budget + (y * param.budget_increase)
    #     #if y < 5 or y > param.planning_window:
    #     #    budget[y] *= 0.25

    capability_streams = param.capability_stream_proportions.shape[0]


    total_planning_budget = np.sum(budget[0:param.planning_window])
    capability_stream_budgets = total_planning_budget * param.capability_stream_proportions
    #initiation_budget = np.ndarray((budget_period, 2))
    #initiation_budget[:, 0] = budget * param.initiation_budget_range[0]
    initiation_budget = budget * param.initiation_max_proportion

    # ongoing_budget = np.ndarray(budget_period)
    ongoing_budget = budget * param.ongoing_max_proportion

    if fuzz:
        (fuzz_before, fuzz_after) = _fuzzification(0.8, 0.01)

        for i in range(min(budget_period, fuzz_before.shape[0])):
            budget[i] *= fuzz_before[i]

        for i in range(min(budget_period - param.planning_window, fuzz_after.shape[0])):
            budget[param.planning_window + i] *= fuzz_after[i]
            # TODO: budget after this window = 0?

    return PortfolioSelectionInstance(random_project_list, budget, capability_stream_budgets, initiation_budget,
                                      ongoing_budget, param.initiation_range, param.planning_window, param.discount_rate, param, identifier)

#TODO: randomize the fuzzification
def _fuzzification(factor=0.7, limit=0.05):
    index = 0
    _initial_fuzz = [factor]
    value = factor * factor
    while value > limit:
        _initial_fuzz.append(value)
        value *= factor

    array_fuzz = np.asarray(_initial_fuzz)

    return 1 - array_fuzz, array_fuzz

# TODO: this seeding means different parameters with similar properties end up with the same projects
def generate_project_files(path, parameter_list, instances=1):
    if not os.path.exists(path):
        os.makedirs(path)

    for param in parameter_list:
        for i in range(instances):
            instance = generate_instance(param, random_seed=i, identifier=i)
            filename = os.path.join(path, f"instance_{param.num_projects}_"
                                          f"{param.planning_window}_{param.base_budget}_{param.budget_increase}_"
                                          f"{param.value_func.value}_{param.cost_distribution.value}_{i}.dat")
            instance.write_to_file(filename)

def get_instances_from_directory(directory):
    result = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".dat"):
            instance_file = os.path.join(directory, filename)
            result += [portfolio_selection_instance_from_file(instance_file)]
    return result
