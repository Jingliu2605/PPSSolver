# Jing Liu
import os
from solvers.gurobi_solver import GurobiSolver
from solvers.gurobi_solver_local import GurobiSolverLocal
import copy
import numpy as np
from problem.test_portfolio_with_repair import build_from_sub_array
import problem.portfolio


def get_gurobi_start_from_ea_solutions(ea_solutions, instance):
    n_start = ea_solutions.ndim
    if n_start == 1:
        # TODO: if n_start > 1
        # start_solutions = np.zeros((instance.num_projects, instance.planning_window), dtype=int)
        start_solutions = {}
        for i in range(instance.num_projects):
            for j in range(instance.planning_window):
                if j == ea_solutions[i] - 1:
                    start_solutions[i, j] = 1
                else:
                    start_solutions[i, j] = 0
    return n_start, start_solutions


def local_search_gurobi(ea_individual, instance, flag, output_dir, t_limitation, i):
    if isinstance(ea_individual, np.ndarray):
        ea_solution = ea_individual
    else:
        ea_solution = ea_individual.get("X")
        ea_solution = np.round(ea_solution).astype(int)
    n_start, start_solutions = get_gurobi_start_from_ea_solutions(ea_solution, instance)
    n_sol = 10
    search_model = 1
    exact_portfolio, gurobi_pool_solutions, pool_fitness, model = run_gurobi_with_seeds_local(instance, i,
                                                                                              time_limitation=t_limitation,
                                                                                              n_start=n_start,
                                                                                              start_solutions=start_solutions,
                                                                                              n_solutions=n_sol,
                                                                                              pool_search_mode=search_model,
                                                                                              flag=flag,
                                                                                              output_dir=output_dir)
    return gurobi_pool_solutions, pool_fitness, model


def run_gurobi_with_seeds_local(instance, i, time_limitation, n_start, start_solutions, n_solutions,
                                pool_search_mode, flag, output_dir):
    log_file = os.path.join(output_dir, f"Dec_Gurobi_{time_limitation}_{i}.log")  #
    exact_solver = GurobiSolverLocal(instance, time_limit=time_limitation, error_threshold=0.01,
                                     num_solutions=n_solutions, pool_search_mode=pool_search_mode, num_start=n_start,
                                     start_solutions=start_solutions, run_flag=flag, output_dir=output_dir)

    exact_portfolio, status, model, exec_time, pool_solutions, pool_fitness = exact_solver.solve(verbose=True,
                                                                                                 log_to_console=True,
                                                                                                 log_file=log_file)
    print(f"Gurobi solution's fitness: {exact_portfolio.value:.3f}")

    return exact_portfolio, pool_solutions, pool_fitness, model


# assuming projects with exclusion and prerequisites are in the same group
def build_sub_instance_from_sub_solution(instance, sub_portfolio, sub_index):
    sub_instance = copy.deepcopy(instance)
    # extract the parameters from sub_portfolio
    sub_instance.budget -= sub_portfolio.cost
    sub_instance.capability_stream_budgets -= sub_portfolio.capability_stream_costs
    sub_instance.ongoing_budget -= sub_portfolio.ongoing_costs
    sub_instance.initiation_budget[:len(sub_portfolio.start_costs)] -= sub_portfolio.start_costs
    sub_instance.num_projects = len(sub_index)
    # obtain the projects in the sub_instance
    # sub_instance.projects = instance.projects[sub_index]
    sub_instance.projects = np.empty(sub_instance.num_projects, dtype=object)
    for i in range(sub_instance.num_projects):
        idx = sub_index[i]
        sub_instance.projects[i] = copy.deepcopy(instance.projects[idx])
        # change the exclusion index
        num_exclusion = len(sub_instance.projects[i].exclusion_list)
        if num_exclusion != 0:
            for j in range(num_exclusion):
                # projects with exclusion and prerequisites are in the same group
                sub_instance.projects[i].exclusion_list[j] = \
                    np.where(sub_index == sub_instance.projects[i].exclusion_list[j])[0]

        # change the prerequisites index
        num_prerequisites = len(sub_instance.projects[i].prerequisite_list)
        if num_prerequisites != 0:
            for j in range(num_prerequisites):
                # projects with exclusion and prerequisites are in the same group
                sub_instance.projects[i].prerequisite_list[j] = \
                    np.where(sub_index == sub_instance.projects[i].prerequisite_list[j])[0]

        # change the successor_list index, the heuristics uses this
        num_successors = len(sub_instance.projects[i].successor_list)
        if num_successors != 0:
            for j in range(num_successors):
                # projects with exclusion and prerequisites are in the same group
                sub_instance.projects[i].successor_list[j] = \
                    np.where(sub_index == sub_instance.projects[i].successor_list[j])[0]

    return sub_instance


def run_gurobi_for_decomposed_problem(instance, sub_index, current_best, t_limitation, output_dir, i):
    num_projects = current_best.shape[0]
    # rest_index = np.setdiff1d(np.arange(num_projects), sub_index)
    # Optimize sub_index
    sub_portfolio, sub_violations = build_from_sub_array(current_best, instance, sub_index)
    sub_instance = build_sub_instance_from_sub_solution(instance, sub_portfolio, sub_index)
    sub_solutions, start_fitness, model = local_search_gurobi(current_best[sub_index], sub_instance, 0,
                                                              output_dir, t_limitation, i)
    current_best[sub_index] = sub_solutions[0]
    current_best_fitness = sub_portfolio.value + start_fitness[0]
    # portfolio, violations = problem.portfolio.build_from_array(current_best, instance)
    return current_best, current_best_fitness


def run_gurobi_for_decomposed_problem_multi_solutions(instance, sub_index, current_best, t_limitation, output_dir, i):
    # num_projects = current_best.shape[0]
    # rest_index = np.setdiff1d(np.arange(num_projects), sub_index)
    sub_portfolio, sub_violations = build_from_sub_array(current_best, instance, sub_index)
    sub_instance = build_sub_instance_from_sub_solution(instance, sub_portfolio, sub_index)
    sub_solutions, start_fitness, model = local_search_gurobi(current_best[sub_index], sub_instance, 0,
                                                              output_dir, t_limitation, i)
    new_x = np.tile(current_best, (len(sub_solutions), 1))
    new_x[:, sub_index] = sub_solutions
    new_fitness = sub_portfolio.value + start_fitness
    # portfolio, violations = problem.portfolio.build_from_array(current_best, instance)
    return new_x, new_fitness


def generate_new_instance_with_changes(instance, pre_solution, current_year=0, omit_projects=None, reduce_budget=0):
    # projects to schedule at current_year
    # Not current_year must > 1
    # Todo: current_year = 1
    if current_year != 0 & current_year != 1:
        implemented_index = list(np.where((current_year > pre_solution) & (pre_solution > 0))[0])
        sub_index = np.setdiff1d(np.arange(instance.num_projects), implemented_index)
        sub_portfolio, sub_violations = build_from_sub_array(pre_solution, instance, sub_index)

        for i in implemented_index:
            # Handle exclusions: Remove any other projects from sub_index in the exclusion list of the current project
            for j in instance.projects[i].exclusion_list:
                instance, sub_index, pre_solution = remove_project_from_instance(instance, sub_index, pre_solution, j)
            # Handle prerequisites: If current project is a prerequisite for another project, remove current project
            # from prerequisite list
            for j in sub_index:
                if i in instance.projects[j].prerequisite_list:
                    instance.projects[j].prerequisite_list = np.delete(instance.projects[j].prerequisite_list, np.where(
                        instance.projects[j].prerequisite_list == i))
    else:
        # Todo: other initialization
        sub_index = list(range(instance.num_projects))

    # omit_projects:
    if omit_projects is not None:
        for i in omit_projects:
            # if projects to omit is in sub_index, remove it
            if i in sub_index:
                instance, sub_index, pre_solution = remove_project_from_instance(instance, sub_index, pre_solution, i)

    if current_year != 0 or omit_projects is not None:
        instance = build_sub_instance_from_sub_solution(instance, sub_portfolio, sub_index)

    if current_year >= 1:
        pre_solution_sub_instance = pre_solution[sub_index]-(current_year - 1)
        pre_solution_sub_instance[pre_solution_sub_instance < 0] = 0
    else:
        pre_solution_sub_instance = pre_solution[sub_index]
    if current_year != 0 & current_year != 1:
        instance.planning_window -= (current_year - 1)
        # instance.num_projects = len(sub_index)
        instance.initiation_budget = instance.initiation_budget.base[current_year-1:]
        instance.ongoing_budget = instance.ongoing_budget.base[current_year - 1:]
        instance.budget = instance.budget.base[current_year - 1:]
        instance.budget_window -= (current_year - 1)
        instance.initiation_range = (0, instance.num_projects)

    if reduce_budget != 0:
        instance.ongoing_budget = reduce_budget*instance.ongoing_budget.base
        instance.initiation_budget = reduce_budget*instance.initiation_budget.base
        instance.budget = reduce_budget*instance.budget.base
        instance.capability_stream_budgets = reduce_budget*instance.capability_stream_budgets.base

    instance.identifier = f"{instance.identifier}_c{current_year}_r{reduce_budget}"
    return instance, pre_solution_sub_instance


# remove the project i from to_optimize, the start time of i set to 0 in the pre_solution
def remove_project_from_instance(instance, to_optimize, pre_solution, i):
    # Stack to keep track of projects to be processed
    stack = [i]

    while stack:
        current_project = stack.pop()

        # Remove current project from to_optimize and set its pre_solution to 0
        to_optimize = np.delete(to_optimize, np.where(to_optimize == current_project))
        pre_solution[current_project] = 0

        # Handle prerequisites: If current project is a prerequisite for another project, add that project to the stack
        for j in to_optimize.copy():
            if current_project in instance.projects[j].prerequisite_list:
                stack.append(j)

        # Handle exclusions: Remove current project from the exclusion list of any other project
        for j in to_optimize.copy():
            if current_project in instance.projects[j].exclusion_list:
                instance.projects[j].exclusion_list = np.delete(instance.projects[j].exclusion_list,
                                                                np.where(instance.projects[
                                                                             j].exclusion_list == current_project))
            if current_project in instance.projects[j].successor_list:
                instance.projects[j].successor_list = np.delete(instance.projects[j].successor_list,
                                                                np.where(instance.projects[
                                                                             j].successor_list == current_project))

    return instance, to_optimize, pre_solution
