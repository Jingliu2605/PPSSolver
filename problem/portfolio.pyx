# cython: boundscheck=False, wraparound=False, profile=True

import array
import pickle

cimport numpy as np
import numpy as np
from cpython cimport array

# TODO: complete refactor to use -1 as unscheduled and 0 - (T-1) as scheduled times
from problem.enums import SchedulingOrder, SchedulingStatus
from problem.project cimport Project
from problem.project import Project
from util import nparray_tostring_helper
from problem.portfolio_selection_instance import PortfolioSelectionInstance
from problem.portfolio_selection_instance cimport PortfolioSelectionInstance


cdef class Portfolio:
    cdef array.array _cost, _result, completed, _stream_costs, _start_costs, _ongoing_costs, _value_over_time, _start_counts
    cdef double _value
    cdef int budget_window
    cdef int planning_window
    cdef double discount_rate
    cdef public int capability_streams
    cdef int _max_time

    @property
    def cost(self):
        return np.array(self._cost, dtype=np.double)

    @property
    def start_costs(self):
        return np.array(self._start_costs, dtype=np.double)

    @property
    def ongoing_costs(self):
        return np.array(self._ongoing_costs, dtype=np.double)

    @property
    def capability_stream_costs(self):
        return np.array(self._stream_costs, dtype=np.double)

    @property
    def result(self):
        return np.array(self._result, dtype=int)

    @property
    def value(self):
        return self._value

    @property
    def value_over_time(self):
        return np.array(self._value_over_time, dtype=np.double)

    def __init__(self, int num_projects, int budget_window, int planning_window, double discount_rate=0.0,
                 int capability_streams=1):

        # this is the completion time for an unscheduled project
        self._max_time = budget_window + 1

        self._result = array.array('i')
        array.resize(self._result, num_projects)
        array.zero(self._result)

        self.completed = array.array('i', [self._max_time] * num_projects)

        self._cost = array.array('d')
        array.resize(self._cost, budget_window)
        array.zero(self._cost)

        self._value_over_time = array.array('d')
        array.resize(self._value_over_time, budget_window)
        array.zero(self._value_over_time)

        self.capability_streams = capability_streams
        self._stream_costs = array.array('d')
        array.resize(self._stream_costs, capability_streams)
        array.zero(self._stream_costs)

        self._start_costs = array.array('d')
        array.resize(self._start_costs, planning_window)
        array.zero(self._start_costs)

        self._ongoing_costs = array.array('d')
        array.resize(self._ongoing_costs, budget_window)
        array.zero(self._ongoing_costs)

        self._start_counts = array.array('i')
        array.resize(self._start_counts, planning_window)
        array.zero(self._start_counts)

        self._value = 0
        self.budget_window = budget_window
        self.planning_window = planning_window
        self.discount_rate = discount_rate

    def __getstate__(self):
        """
        Define how to pickle a portfolio.
        :return:
        """
        state = dict()
        state['cost'] = self._cost.tobytes()
        state['result'] = self._result.tobytes()
        state['completed'] = self.completed.tobytes()
        state['stream_costs'] = self._stream_costs.tobytes()
        state['start_costs'] = self._start_costs.tobytes()
        state['start_counts'] = self._start_counts.tobytes()
        state['ongoing_costs'] = self._ongoing_costs.tobytes()
        state['value_over_time'] = self._value_over_time.tobytes()

        state['value'] = self._value
        state['budget_window'] = self.budget_window
        state['planning_window'] = self.planning_window
        state['discount_rate'] = self.discount_rate
        state['capability_streams'] = self.capability_streams
        state['max_time'] = self._max_time

        return state

    def __setstate__(self, state):
        """
        Define how to unpickle a portfolio.
        :return:
        """

        self._cost = array.array('d')
        self._cost.frombytes(state['cost'])
        self._result = array.array('i')
        self._result.frombytes(state['result'])
        self.completed = array.array('i')
        self.completed.frombytes(state['completed'])
        self._stream_costs = array.array('d')
        self._stream_costs.frombytes(state['stream_costs'])
        self._start_costs = array.array('d')
        self._start_costs.frombytes(state['start_costs'])
        self._ongoing_costs = array.array('d')
        self._ongoing_costs.frombytes(state['ongoing_costs'])
        self._start_counts = array.array('i')
        self._start_counts.frombytes(state['start_counts'])
        self._value_over_time = array.array('d')
        self._value_over_time.frombytes(state['value_over_time'])

        self._value = state['value']
        self.budget_window = state['budget_window']
        self.planning_window = state['planning_window']
        self.discount_rate = state['discount_rate']
        self.capability_streams = state['capability_streams']
        self._max_time = state['max_time']

    cpdef scheduled(self, int index):
        return self._result.data.as_ints[index] > 0

    cpdef start_time(self, int index):
        return self._result.data.as_ints[index]

    #cpdef start_count(self, int time):
    #    # TODO: this assumes that time is 1-based indexing
    #    return self._start_counts.data.as_ints[time - 1]

    cpdef add_to_portfolio(self, int candidate_p, int t, Project project):
        """
        Add a project to the portfolio and update associated cost and value tracking information.
        
        :param candidate_p: The index of the project to add.
        :param t: The year in which the project starts.
        :param project: The Project object, which provides the necessary information.
        :return: 
        """
        cdef int t2, i, abs_t
        cdef double discounted_value

        # add the start and completion times to the respective arrays
        self._result.data.as_ints[candidate_p] = t
        self.completed.data.as_ints[candidate_p] = t + project.duration
        self._start_counts.data.as_ints[t - 1] += 1

        # add the cost of the first year to the start cost array
        self._start_costs[t - 1] += project.cost_raw.data.as_doubles[0]

        # for each year that the project runs, add the cost to the associated running total for its respective stream
        # NOTE: this is only calculated for the years within the planning window
        for t2 in range(min(project.duration, self.planning_window - t + 1)):
            # include the discount rate based on the year of start
            # add the value for each year in which the project runs during the planning window
            #self._value += project.value[t2] / ((1 + t - 1) ** self.discount_rate)

            # add the associated costs for the capability stream only within the planning horizon
            self._stream_costs[project.capability_stream] += project.cost_raw.data.as_doubles[t2]

        # for each year that the project runs, add its cost, value, and the ongoing cost for each year after the first
        for t2 in range(project.duration):
            abs_t = t+t2-1
            self._cost.data.as_doubles[abs_t] += project.cost_raw.data.as_doubles[t2]
            discounted_value = self.time_discounted(project.value[t2], abs_t)
            self._value += discounted_value # / ((1 + self.discount_rate) ** (t + t2 - 1))
            self._value_over_time[abs_t] += discounted_value # / ((1 + self.discount_rate) ** (t + t2 - 1))
            if t2 > 0 :
                self._ongoing_costs[abs_t] += project.cost_raw.data.as_doubles[t2]

    cdef inline time_discounted(self, value, t):
        return value/((1 + self.discount_rate) ** t)

    cpdef remove_from_portfolio(self, int candidate_p, Project project):
        """
        Remove a project from the portfolio and update associated cost and value tracking information.
        
        :param candidate_p: The index of the project to remove.
        :param project: The Project object, which provides the necessary information.
        :return: 
        """

        cdef int t = self._result.data.as_ints[candidate_p]
        if t == 0:
            return
        cdef int t2, i, abs_t
        cdef double discounted_value

        self._result.data.as_ints[candidate_p] = 0
        self.completed.data.as_ints[candidate_p] = self._max_time
        self._start_counts.data.as_ints[t - 1] -= 1

        self._start_costs[t - 1] -= project.cost_raw.data.as_doubles[0]

        for t2 in range(min(project.duration, self.planning_window - t + 1)):
            #self._value -= project.value[t2] / ((1 + t - 1) ** self.discount_rate)
            self._stream_costs[project.capability_stream] -= project.cost_raw.data.as_doubles[t2]
        #self._value -= project.value

        for t2 in range(project.duration):
            # TODO: prevent recalculation
            abs_t = t+t2-1
            self._cost.data.as_doubles[abs_t] -= project.cost_raw.data.as_doubles[t2]
            discounted_value = self.time_discounted(project.value[t2], abs_t)
            self._value -= discounted_value
            self._value_over_time[abs_t] -= discounted_value
            if t2 > 0 :
                self._ongoing_costs[abs_t] -= project.cost_raw.data.as_doubles[t2]

    cpdef add_if_feasible(self, int index, int t, Project project, PortfolioSelectionInstance instance):
        """
        Add a project to the portfolio, if it is feasible with respect to the provided problem instance.
        :param index: 
        :param t: 
        :param project: 
        :param instance: 
        :return: 
        """
        if self.feasibility_check(project, t, instance):
            self.add_to_portfolio(index, t, project)

    cpdef exclusion_check(self, Project project):
        cdef int exclusion, index

        for index in range(project.exclusion_list.shape[0]):
            exclusion = project.exclusion_list[index]
            if self.completed.data.as_ints[exclusion] != self._max_time:
                return False

        return True

    cpdef prerequisite_check(self, Project project, int t):
        cdef int prerequisite, index

        for index in range(project.prerequisite_list.shape[0]):
            prerequisite = project.prerequisite_list[index]
            if self.completed.data.as_ints[prerequisite] > t:
                return False

        return True

    cpdef completion_window_check(self, Project project, int t):
        cdef int project_completion, completion_start, completion_end

        # no constraints on completion time
        if project.completion_window is None:
            return True

        project_completion = t + project.duration

        # set to minimum possible value if not given
        if project.completion_window[0] is None:
            completion_start = -1
        else:
            completion_start = project.completion_window[0]

        #set to maximum possible value if not given
        if project.completion_window[1] is None:
            completion_end = project_completion + 1 # hack to ensure that comparison returns true
        else:
            completion_end = project.completion_window[1]

        return completion_start <= project_completion <= completion_end


    cpdef feasibility_check(self, Project project, int t, PortfolioSelectionInstance instance):
        """
        Determines if a project can be feasibly scheduled at time t, with respect to the provided problem instance.
        :param project: 
        :param t: 
        :param instance: 
        :return: 
        """
        cdef int t2, stream_end
        cdef double stream_cost

        # ensure that we are able to initiate a new project at this time instance
        if self._start_counts.data.as_ints[t - 1] + 1 > instance.initiation_range[1]:
            return False

        # ensure no mutual exclusions have also been selected
        if not self.exclusion_check(project):
            return False

         # ensure that prerequisites have been satisfied
        if not self.prerequisite_check(project, t):
            return False

        # ensure project would be completed within the required window
        # if not self.completion_window_check(project, t):
        #    return False

        # check that start cost does not exceed feasible maximum
        if self._start_costs.data.as_doubles[t - 1] + project.cost_raw.data.as_doubles[0] \
                > instance.initiation_budget[t - 1]:
            return False

        stream_end = min(project.duration, self.planning_window - t + 1)
        stream_cost = self._stream_costs.data.as_doubles[project.capability_stream]

        #cdef np.double_t[:] capability_stream_budgets = instance.capability_stream_budgets
        #cdef np.double_t[:] budget = instance.budget
        #cdef np.double_t[:] ongoing_budget = instance.ongoing_budget

        # check that cost is less than budget
        for t2 in range(project.duration):
            if t2 < stream_end:
                stream_cost += project.cost_raw.data.as_doubles[t2]
                if stream_cost > instance.capability_stream_budgets[project.capability_stream]:
                    return False

            if self._cost.data.as_doubles[t + t2 - 1] + project.cost_raw.data.as_doubles[t2] > instance.budget[t + t2 - 1]:
                return False

            if t2 > 0:
                if self._ongoing_costs.data.as_doubles[t+t2-1] + project.cost_raw.data.as_doubles[t2] > instance.ongoing_budget[t+t2-1]:
                    return False

        return True

    cpdef constraint_violations(self, PortfolioSelectionInstance instance):

        cdef Project[:] projects = instance.projects

        cdef int i, t, prerequisite, exclusion, index, prerequisite_viols, exclusion_viols

        prerequisite_viols = 0  # prerequisite violations
        exclusion_viols = 0  # mutual exclusion violations
        # loop through each time step in start_times and calculate project properties
        for i in range(len(self._result)):
            t = self._result.data.as_ints[i]
            if t <= 0: # if t <= 0, project is not implemented -> skip
                continue

            # check that prerequisites are completed
            for index in range(projects[i].prerequisite_list.shape[0]):
                prerequisite = projects[i].prerequisite_list[index]
                if self.completed.data.as_ints[prerequisite] > t:
                    prerequisite_viols += 1

            # ensure no mutual exclusions have also been selected
            for index in range(projects[i].exclusion_list.shape[0]):
                exclusion = projects[i].exclusion_list[index]
                if self._result.data.as_ints[exclusion] > 0:
                    exclusion_viols += 1

        # TODO: add initiation range violation check
        # for t in range(self.planning_window):
        #    pass


        budget_viols = self._generic_budget_violation(instance.budget, self._cost, self.budget_window)
        cs_viols = self._generic_budget_violation(instance.capability_stream_budgets, self._stream_costs,
                                                  self.capability_streams)
        initiation_viols = self._generic_budget_violation(instance.initiation_budget, self._start_costs,
                                                          self.planning_window)
        ongoing_viols = self._generic_budget_violation(instance.ongoing_budget, self._ongoing_costs,
                                                       self.budget_window)

        all_viols = np.concatenate((budget_viols, cs_viols, initiation_viols, ongoing_viols,
                                    [prerequisite_viols, exclusion_viols]))

        return {
            "budget_viols": budget_viols,
            "stream_viols": cs_viols,
            "initiation_viols": initiation_viols,
            "ongoing_violations": ongoing_viols,
            "prereq_viols": prerequisite_viols,
            "exclusion_viols": exclusion_viols,
            "all_viols": all_viols
        }

    cpdef _generic_budget_violation(self, const np.double_t[:] budget, const np.double_t[:] cost, int periods):
        cdef int t
        cdef double c, b
        # check that the budget is satisfied at each time instance
        viols = np.zeros(periods)  # budget violations
        for t in range(periods):
            c = cost[t]
            b = budget[t]
            viols[t] = 0 if c <= b else c - b #max(self._cost.data.as_doubles[t] - budget[t], 0)
        return viols

    cpdef budget_violation_reduction(self, int index, Project project, np.double_t[:] budget):
        cdef int start = self._result.data.as_ints[index]
        cdef int t, t2, abs_index
        cdef double b, c, reduction, viol_amount, time_cost

        # if project is not scheduled, return 0
        if start == 0:
            return 0

        budget_viols = np.zeros(self.budget_window)  # budget violations
        for t in range(self.budget_window):
            c = self._cost.data.as_doubles[t]
            b = budget[t]
            budget_viols[t] = 0 if c <= b else c - b #max(self._cost.data.as_doubles[t] - budget[t], 0)

        reduction = 0
        for t2 in range(project.duration):
            abs_index = start + t2 - 1
            viol_amount = budget_viols[abs_index]
            time_cost = project.cost_raw.data.as_doubles[t2] #project.cost_at_time(t2)
            if viol_amount == 0:             # no violation, move to next time step
                continue
            elif time_cost > viol_amount: # if cost is greater than violation amount, increase by violation amount
                reduction += viol_amount
            else: # cost <= violation amount, subtract the cost at this time
                reduction += time_cost

        return reduction

    cpdef add_earliest_feasible(self, int index, Project project, PortfolioSelectionInstance instance):
        cdef int time
        time = self.find_earliest(project, instance)
        if time > 0:
            self.add_to_portfolio(index, time, project)

    cpdef add_latest_feasible(self, int index, Project project, PortfolioSelectionInstance instance):
        cdef int time
        time = self.find_latest(project, instance)
        if time > 0:
            self.add_to_portfolio(index, time, project)

    cpdef int find_earliest(self, Project project, PortfolioSelectionInstance instance):
        cdef int t
        for t in range(1, instance.planning_window + 1):
            if self.feasibility_check(project, t, instance):
                return t

        return -1  # this indicates that a project can not be scheduled without violating constraints

    #TODO: refactor feasibility test to match this...
    cpdef feasibility_status(self, Project project, int t, PortfolioSelectionInstance instance):
        cdef int t2

        # ensure no mutual exclusions have also been selected
        if not self.exclusion_check(project):
            return SchedulingStatus.EXCLUSION_VIOL

         # ensure that prerequisites have been satisfied
        if not self.prerequisite_check(project, t):
            return SchedulingStatus.PREREQUISITE_VIOL

        # ensure project would be completed within the required window
        if not self.completion_window_check(project, t):
            return SchedulingStatus.COMPLETION_WINDOW_VIOL

        # check that cost is less than budget
        for t2 in range(project.duration):
            if self._cost.data.as_doubles[t + t2 - 1] + project.cost_raw.data.as_doubles[t2] > instance.budget[t + t2 - 1]:
                return SchedulingStatus.BUDGET_VIOL

        # TODO: add capability stream budgets
        # TODO: add start cost feasibility
        # TODO: add ongoing cost feasibility

        return SchedulingStatus.SUCCESS

    cpdef find_latest(self, Project project, PortfolioSelectionInstance instance):
        cdef int t
        for t in range(instance.planning_window, 0, -1):
            if self.feasibility_check(project, t, instance):
                return t

        return -1  # this indicates that a project can not be scheduled without violating constraints

    def find_random_feasible(self, project, instance):
        time = np.arange(instance.planning_window)
        np.random.shuffle(time)
        for t in time:
            if self.feasibility_check(project, t, instance):
                return t

        return -1

    def start_count(self, budget_window):
        start_counts = {}
        for i in range(self._max_time):
            start_counts[i] = 0

        for i in self._result:
            start_counts[i] += 1

        return start_counts

    def continuing_count(self, budget_window, projects):
        continuing_counts = {}
        for i in range(self._max_time):
            continuing_counts[i] = 0

        for p in range(len(projects)):
            time = self._result[p]
            if time > 0: # exclude unscheduled projects
                project = projects[p]
                # start at 1 to exclude project start time
                for j in range(1, project.duration):
                    continuing_counts[time + j] += 1

        return continuing_counts

    cpdef costs_by_category(self, Project[:] projects):
        cdef int p, time, j

        start_costs = np.zeros(len(self._cost), dtype=float)
        continuing_costs = np.zeros(len(self._cost), dtype=float)

        # TODO: make this more efficient
        for p in range(projects.shape[0]):
            time = self._result.data.as_ints[p]
            if time > 0: # exclude unscheduled projects
                project = projects[p]
                costs = project.cost_raw
                start_costs[time-1] += costs[0]
                # start at 1 to exclude project start time
                for j in range(1, project.duration):
                    continuing_costs[time + j - 1] += costs[j]

        return start_costs, continuing_costs

    cpdef cost_value_by_stream(self, PortfolioSelectionInstance instance):
        cdef Project[:] projects = instance.projects
        cdef int p, t, j, num_streams

        num_streams = instance.capability_stream_budgets.shape[0]

        costs = np.zeros((num_streams, instance.budget_window), dtype=np.double)
        values = np.zeros((num_streams, instance.budget_window), dtype=np.double)

        # TODO: make this more efficient
        for p in range(projects.shape[0]):
            t = self._result.data.as_ints[p]
            if t > 0: # exclude unscheduled projects
                project = projects[p]
                cap_stream = project.capability_stream
                for j in range(project.duration):
                    costs[cap_stream, t + j - 1] += project.cost_at_time(j)
                    values[cap_stream, t + j - 1] += project.value_at_time(j)

        return costs, values

    def write_to_file(self, filename, instance):
        start_costs, continuing_costs = self.costs_by_category(instance.projects)
        violations = self.constraint_violations(instance)
        with open(filename, "w") as file:
            file.write(f"Result: {nparray_tostring_helper(self.result)}\n")
            file.write(f"Value: {self._value}\n")
            file.write(f"Costs: {nparray_tostring_helper(self.cost)}\n")
            file.write(f"Starting Costs: {nparray_tostring_helper(start_costs)}\n")
            file.write(f"Continuing Costs: {nparray_tostring_helper(continuing_costs)}\n")
            file.write(f"Budget Violations: {nparray_tostring_helper(violations['budget_viols'])}\n")
            file.write(f"Prerequisite Violations: {violations['prereq_viols']}\n")
            file.write(f"Exclusion Violations: {violations['exclusion_viols']}\n")
            # TODO: add new budget considerations to file

    def write_to_pickle(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def portfolio_from_pickle(filename):
    with open(filename, "rb") as f:
        portfolio = pickle.load(f)
    return portfolio

# TODO: how to add kwargs to accep
#  t p_random?
# TODO: this doesn't guarantee that we can rebuild the portfolio after optimization...
cpdef build_from_permutation(const np.int_t[:] perm, PortfolioSelectionInstance instance, scheduling_order):
    cdef int t, j, index
    cdef Portfolio portfolio = Portfolio(instance.num_projects, instance.budget_window, instance.planning_window,
                                         instance.discount_rate, instance.capability_stream_budgets.shape[0])

    if scheduling_order is SchedulingOrder.EARLIEST:
        portfolio = _schedule_earliest(perm, instance)
    elif scheduling_order is SchedulingOrder.LATEST:
        portfolio = _schedule_latest(perm, instance)
    elif scheduling_order is SchedulingOrder.STOCHASTIC_EARLIEST:
        # kwargs.setdefault('p_random', 0.1)
        portfolio = _schedule_earliest_stochastic(perm, instance, 0.01)  # 0.05
    #elif scheduling_order is SchedulingOrder.FB:
    #    portfolio = _forward_backward_scheduling(perm, projects, budget, planning_window)
    else:
        raise ValueError("Unknown scheduling order for Portfolio Permutation Problem")

    return portfolio

cpdef _schedule_earliest(const np.int_t[:] perm, PortfolioSelectionInstance instance):
    cdef int t, j, index
    cdef int num_projects = perm.shape[0]
    #cdef int budget_window = budget.shape[0]
    cdef Portfolio portfolio = Portfolio(num_projects, instance.budget_window, instance.planning_window,
                                         instance.discount_rate, instance.capability_stream_budgets.shape[0])

    for index in range(num_projects):
        j = perm[index]
        portfolio.add_earliest_feasible(j, instance.projects[j], instance)

    return portfolio

cpdef schedule_earliest_stochastic_limited(const np.int_t[:] perm, PortfolioSelectionInstance instance, double p_random, int feas_range):
    cdef int t, j, index
    cdef int num_projects = perm.shape[0]
    #cdef int budget_window = budget.shape[0]
    cdef Portfolio portfolio = Portfolio(num_projects, instance.budget_window, instance.planning_window,
                                         instance.discount_rate, instance.capability_stream_budgets.shape[0])

    feas_vector = np.zeros(instance.planning_window, dtype=bool)
    for index in range(num_projects):
        j = perm[index]

        if np.random.random() < p_random:  # randomly select from the feasible start times
            for t in range(1, instance.planning_window + 1):
                feas_vector[t-1] = portfolio.feasibility_check(instance.projects[j], t, instance)

            # check to ensure that there is a feasible start time
            feas_times = feas_vector.nonzero()[0]
            if feas_times.shape[0] == 0:
                continue
            if feas_times.shape[0] > feas_range:
                feas_times = feas_times[0:feas_range]
            # add 1 given that feas_vector uses 0-based, while portfolio expects 1-based
            start_time = np.random.choice(feas_times) + 1
            portfolio.add_to_portfolio(j, start_time, instance.projects[j])
        else:  # schedule the project at its earliest feasible start time
            portfolio.add_earliest_feasible(j, instance.projects[j], instance)

    return portfolio


cpdef _schedule_earliest_stochastic(const np.int_t[:] perm, PortfolioSelectionInstance instance, double p_random):
    cdef int t, j, index
    cdef int num_projects = perm.shape[0]
    #cdef int budget_window = budget.shape[0]
    cdef Portfolio portfolio = Portfolio(num_projects, instance.budget_window, instance.planning_window,
                                         instance.discount_rate, instance.capability_stream_budgets.shape[0])

    feas_vector = np.zeros(instance.planning_window, dtype=bool)
    for index in range(num_projects):
        j = perm[index]

        if np.random.random() < p_random:  # randomly select from the feasible start times
            for t in range(1, instance.planning_window + 1):
                feas_vector[t-1] = portfolio.feasibility_check(instance.projects[j], t, instance)

            # check to ensure that there is a feasible start time
            feas_times = feas_vector.nonzero()[0]
            if feas_times.shape[0] == 0:
                continue
            if feas_times.shape[0] > 4:   # Jing Liu
                feas_times = feas_times[0:4]
            # add 1 given that feas_vector uses 0-based, while portfolio expects 1-based
            start_time = np.random.choice(feas_times) + 1
            portfolio.add_to_portfolio(j, start_time, instance.projects[j])
        else:  # schedule the project at its earliest feasible start time
            portfolio.add_earliest_feasible(j, instance.projects[j], instance)

    return portfolio

cpdef _schedule_latest(const np.int_t[:] perm, PortfolioSelectionInstance instance):

    cdef int t, j, index
    cdef int num_projects = perm.shape[0]
    #cdef int budget_window = budget.shape[0]
    cdef Portfolio portfolio = Portfolio(num_projects, instance.budget_window, instance.planning_window,
                                         instance.discount_rate,
                                         instance.capability_stream_budgets.shape[0])

    for index in range(num_projects):
        j = perm[index]
        t = portfolio.add_latest_feasible(j, instance.projects[j], instance)

    return portfolio

cpdef build_from_array(const np.int_t[:] x, PortfolioSelectionInstance instance):
    cdef int i, t, prereq, exclusion, num_projects, budget_window
    """
    This is a more efficient manner to construct a portfolio and calculate the violations from an array
    :param x: 
    :param projects: 
    :param budget: 
    :param planning_window: 
    :return: 
    """
    cdef int planning_window = instance.planning_window
    num_projects = x.shape[0]
    #budget_window = budget.shape[0]
    portfolio = Portfolio(num_projects, instance.budget_window, planning_window, instance.discount_rate,
                          instance.capability_stream_budgets.shape[0])

    # ignore projects that are not scheduled or are scheduled outside the planning window
    for i in range(num_projects):
        if 0 < x[i] <= planning_window:
            #TODO: should this be x[i] for the time?
            portfolio.add_to_portfolio(i, x[i], instance.projects[i])

    violations = portfolio.constraint_violations(instance)

    return portfolio, violations

# Jing Liu
cpdef build_from_array_and_repair(np.int_t[:] phenotype, x,  PortfolioSelectionInstance instance, int real_flag):
    cdef int i, t, prereq, exclusion, num_projects, budget_window
    """
    This is a more efficient manner to construct a portfolio and calculate the violations from an array
    :param x: 
    :param projects: 
    :param budget: 
    :param planning_window: 
    :return: 
    """
    cdef int planning_window = instance.planning_window
    num_projects = phenotype.shape[0]
    #budget_window = budget.shape[0]
    portfolio = Portfolio(num_projects, instance.budget_window, planning_window, instance.discount_rate,
                          instance.capability_stream_budgets.shape[0])

    # ignore projects that are not scheduled or are scheduled outside the planning window
    for i in range(num_projects):
        if 0 < phenotype[i] <= planning_window:
            if portfolio.feasibility_check(instance.projects[i], phenotype[i], instance):
                portfolio.add_to_portfolio(i, phenotype[i], instance.projects[i])
            else:
                x[i] = 0
                phenotype[i] = 0
                # Todo: add_earliest_feasible
        else:
            x[i] = 0
            phenotype[i] = 0

    # violations = portfolio.constraint_violations(instance)
    violations = 0
    return portfolio, violations, x, phenotype