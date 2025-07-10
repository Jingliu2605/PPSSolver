import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from problem.portfolio import Portfolio

import os


class GurobiSolverLocal:

    def __init__(self, instance, time_limit=None, error_threshold=None, num_solutions=None, pool_search_mode=None,
                 num_start=None, start_solutions=None, run_flag=0, output_dir=None):
        self.projects = instance.projects
        self.budget = instance.budget
        self.capability_stream_budgets = instance.capability_stream_budgets
        self.initiation_budget = instance.initiation_budget
        self.initiation_range = instance.initiation_range
        self.ongoing_budget = instance.ongoing_budget
        self.num_projects = len(self.projects)
        self.budget_window = len(self.budget)
        self.planning_window = instance.planning_window
        self.time_limit = time_limit
        self.num_solutions = num_solutions
        self.PoolSearchMode = pool_search_mode
        self.error_threshold = error_threshold
        self.exec_time = 0
        self.discount_rate = instance.discount_rate
        self.num_start = num_start
        self.start_solutions = start_solutions
        self.run_flag = run_flag  # if the model is known, run_flag = 1; otherwise, 0
        self.output_dir = output_dir

    def _initialize(self):
        # TODO: should this be int?
        costs = np.zeros((self.num_projects, self.budget_window))
        durations = np.zeros(self.num_projects)
        values = [None] * self.num_projects

        prerequisites = []
        exclusions = []

        for i in range(self.num_projects):
            project = self.projects[i]
            values[i] = project.value
            costs[i, 0:project.duration] = project.cost
            durations[i] = project.duration
            for p in project.prerequisite_list:
                prerequisites += [(p, i)]

            for e in project.exclusion_list:
                exclusions += [(e, i)]

        return self.planning_window, durations, values, costs, prerequisites, exclusions

    def _setup(self, time_periods, durations, values, costs, budget, capability_stream_budgets, initiation_budget,
               initiation_range, ongoing_budget, bp, prereqs, me, log_to_console):
        start_time = time.perf_counter()
        """_setup -- model for the portfolio selection and scheduling problem
        Parameters:
            - projects: set of projects
            - time_periods: number of periods
            - durations[j]: duration of project j
            - values[t]: value of project j.
            - costs[j,t]: cost for project j during period t (after project starts)
            - budget[t]: budget at time t
            - bp: budget period
            - prereqs: list of tuples (i, j) where i must be completed before j can be scheduled
            - me: list of tuples (i, j) where at most one of i or j can be selected
        Returns a model, ready to be solved.
        """
        model = gp.Model("Portfolio Selection and Scheduling")
        print("\tBuilding model ... ", end="")
        model.Params.outputFlag = log_to_console  # hide Gurobi output

        x = model.addVars(self.num_projects, time_periods, vtype=GRB.BINARY, name="x")

        # constraint: project can only be selected once
        model.addConstrs((x.sum(j, '*') <= 1 for j in range(self.num_projects)), "SelectOnce")

        # constraint: budget in each time period
        for t in range(bp):
            # TODO must implement this constraint properly
            model.addConstr(
                gp.quicksum(
                    costs[j, t - t2] * x[j, t2] for j in range(self.num_projects) for t2 in range(self.planning_window)
                    if 0 <= (t - t2) < durations[j]) <= budget[t], f"Budget({t})")

        # ongoing project budget constraint
        for t in range(bp):
            model.addConstr(
                gp.quicksum(
                    costs[j, t - t2] * x[j, t2] for j in range(self.num_projects) for t2 in
                    range(self.planning_window)
                    if 1 <= (t - t2) < durations[j]) <= ongoing_budget[t], f"OngoingBudget({t})")

        # ensure that capability stream budgets are adhered to
        for cs in range(capability_stream_budgets.shape[0]):
            # TODO this can probably be simplified a bit
            model.addConstr(
                gp.quicksum(
                    costs[j, t - t2] * x[j, t2]
                    for t in range(self.planning_window)
                    for j in range(self.num_projects)
                    for t2 in range(self.planning_window)
                    if 0 <= (t - t2) < durations[j]
                    and self.projects[j].capability_stream == cs)
                <= capability_stream_budgets[cs], f"Stream({cs})")

        for t in range(time_periods):
            # ensure that budget for initiating projects is respected
            model.addConstr(gp.quicksum(x[i, t] * costs[i, 0]
                                        for i in range(self.num_projects))
                            <= initiation_budget[t], f"InitiationBudget({t})")

            # TODO: add minimum range
            # ensure that the number of projects starting in any period is within the allowable range
            model.addConstr(gp.quicksum(x[i, t] for i in range(self.num_projects)) <=
                            initiation_range[1], f"InitiationRange({t})")

        # constraint: mutual exclusion
        for (i, j) in me:
            model.addConstr(gp.quicksum(x[i, t] + x[j, t] for t in range(time_periods)) <= 1, "MutualExclusions")

        # constraint: prerequisites
        for (i, j) in prereqs:
            # if j is selected, i must be selected
            model.addConstr(x.sum(j, '*') <= x.sum(i, '*'))

            # j must start after completion of i, if j is selected
            # TODO: sum(j, '*') is a major hack to only enforce this if j is selected, otherwise it prevents i from being selected without j
            model.addConstr(gp.quicksum(t * x[j, t] for t in range(time_periods)) >=
                            gp.quicksum((t + durations[i]) * x[i, t] for t in range(time_periods)) * x.sum(j, '*'),
                            "Prerequisite")

        # add values for each year that a project is running
        model.setObjective(gp.quicksum(
            (values[j][t - t2] / ((1 + self.discount_rate) ** t)) * x[j, t2]  # TODO: verify discount factor
            for j in range(self.num_projects)
            for t in range(bp)
            for t2 in range(time_periods)
            if 0 <= (t - t2) < durations[j]), GRB.MAXIMIZE)

        # set a time limit for the solver, if applicable
        if self.time_limit is not None:
            model.Params.timeLimit = self.time_limit

        # model.Params.solutionLimit = 5000
        if self.error_threshold is not None:
            model.Params.MIPGap = self.error_threshold

        # set the number of solutions
        if self.num_solutions is not None:
            model.Params.PoolSolutions = self.num_solutions
        if self.PoolSearchMode is not None:
            model.Params.PoolSearchMode = self.PoolSearchMode
        if self.num_start is not None:
            model.NumStart = self.num_start
        model.params.Method = 2
        model.params.Threads = 20
        # model.params.MIPFocus = 1
        end_time = time.perf_counter()

        print(f"Done!\n\tModel building took {end_time - start_time:0.1f}s")

        return model, x

    def solve(self, verbose=False, log_to_console=False, log_file=None):

        (time_periods, durations, values, costs, prerequisites, exclusions) = self._initialize()

        if verbose:
            print()
            print("##### Gurobi Solver #####")
            # print(f"Total Value: {values.sum()}")

        # model_dir = os.path.join(self.output_dir, "GurobiLocal")
        #
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)

        # gurobi_model_file = os.path.join(model_dir, "gurobimodel.mps")
        # gurobi_attr_file = os.path.join(model_dir, "gurobiattr.attr")
        # gurobi_hnt_file = os.path.join(model_dir, "gurobihnt.attr")
        # if self.run_flag == 0:

        model, variables = self._setup(time_periods, durations, values, costs, self.budget,
                                       self.capability_stream_budgets, self.initiation_budget,
                                       self.initiation_range,
                                       self.ongoing_budget, self.budget_window, prerequisites, exclusions,
                                       log_to_console)
        model.Params.displayInterval = 10
        model.Params.logToConsole = log_to_console
        # model.write(gurobi_model_file)
        # model.write(gurobi_attr_file)
        # model.write(gurobi_hnt_file)
        # else:
        #     model = gp.read(gurobi_model_file)
        #     # set a time limit for the solver, if applicable
        #     if self.time_limit is not None:
        #         model.Params.timeLimit = self.time_limit
        #
        #     # model.Params.solutionLimit = 5000
        #     if self.error_threshold is not None:
        #         model.Params.MIPGap = self.error_threshold
        #
        #     # set the number of solutions
        #     if self.num_solutions is not None:
        #         model.Params.PoolSolutions = self.num_solutions
        #     if self.PoolSearchMode is not None:
        #         model.Params.PoolSearchMode = self.PoolSearchMode
        #     if self.num_start is not None:
        #         model.NumStart = self.num_start
        #     model.read(gurobi_attr_file)

        model.update()
        variables = model.getVars()

        if log_file is not None:
            # empty the log file, if it exists
            open(log_file, 'w').close()
            model.Params.logFile = log_file

        start_time = time.perf_counter()
        # TODO: if many start_solutions
        if self.start_solutions is not None:
            for (j, t) in self.start_solutions:
                # variables[j, t].Start = self.start_solutions[j, t]
                variables[j * self.planning_window + t].Start = self.start_solutions[j, t]

        # presolve_model = model.presolve()
        # relaxed_presolve_model = presolve_model.relax()
        # relaxed_presolve_model.optimize()

        # model.reset()
        model.optimize()
        end_time = time.perf_counter()
        self.exec_time = end_time - start_time
        status = model.Status
        # GRB.Optimal -> best found
        # GRB.TIME_LIMIT -> time limit reached

        if verbose:
            print(f"Fitness: {model.objVal:.3f}")
            print(f"Time: {self.exec_time:.3E}s")
            print(f"Status: {status}")

        if self.num_solutions is not None:
            pool_solutions = np.ndarray((model.SolCount, self.num_projects), dtype=int)
            suboptimal = np.zeros(model.SolCount, dtype=np.double)
            for i in range(model.SolCount):
                model.Params.SolutionNumber = i
                solution = model.getAttr('xn', variables)
                portfolio = get_portfolio_from_gurobi_solution(self, solution)
                if i == 0:
                    best_portfolio = portfolio
                pool_solutions[i] = portfolio.result
                suboptimal[i] = model.PoolObjVal
        else:
            solution = model.getAttr('x', variables)
            # generate a portfolio based on the Gurobi solution
            best_portfolio = get_portfolio_from_gurobi_solution(self, solution)
            pool_solutions = best_portfolio.result
            suboptimal = best_portfolio.value

        return best_portfolio, status, model, self.exec_time, pool_solutions, suboptimal


def get_portfolio_from_gurobi_solution(self, solution):
    # generate a portfolio based on the Gurobi solution
    portfolio = Portfolio(self.num_projects, self.budget_window, self.planning_window, self.discount_rate,
                          capability_streams=self.capability_stream_budgets.shape[0])
    # for (j, t) in solution:
    #     # regardless of whether time limit reached, this is best known
    #     if solution[j, t] > 0.5:
    #         # add 1 for time as Gurobi uses 0-based time indexes
    #         portfolio.add_to_portfolio(j, t + 1, self.projects[j])
    for i in range(len(solution)):
        j = i // self.planning_window
        t = i % self.planning_window
        if solution[i] > 0.5:
            #         # add 1 for time as Gurobi uses 0-based time indexes
            portfolio.add_to_portfolio(j, t + 1, self.projects[j])

    return portfolio


