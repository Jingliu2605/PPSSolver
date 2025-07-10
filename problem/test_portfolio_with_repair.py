# Jing Liu
import numpy as np

from pymoo.model.problem import Problem
from problem.portfolio import Portfolio
import copy
from problem.portfolio import build_from_permutation, build_from_array, portfolio_from_pickle


class TestPortfolioProblemWithRepair(Problem):

    def __init__(self, instance, pause_event=None, stop_event=None, real_flag=0, p_random=0.5, decomposed=False, **kwargs):
        # No constraints as project are implemented at feasible time
        super().__init__(n_var=len(instance.projects), n_obj=1, n_constr=0, xl=0,
                         xu=instance.planning_window, type_var=int, **kwargs)
        self.instance = instance
        self.real_flag = real_flag
        self.p_random = p_random
        self.decomposed = decomposed
        self.pause_event = pause_event
        self.stop_event = stop_event

    def _evaluate(self, x, out, *args, **kwargs):
        kwargs.setdefault('decomposed', False)
        decomposed = kwargs['decomposed']
        if decomposed:
            dim_index = kwargs['dim_index']
            best_now = kwargs['best_now']
            best_phenotype = np.round(best_now).astype(int)
            sub_portfolio, violations = build_from_sub_array(best_phenotype, self.instance, dim_index)

        count = x.shape[0]
        fits = np.zeros(count, dtype=np.double)
        results = [None] * count
        phenotypes = [None] * count

        for i in range(count):
            phenotype = np.round(x[i]).astype(int)
            if decomposed:
                portfolio, violations, new_x, new_phenotype = (
                    build_from_sub_array_and_repair(phenotype, x[i], self.instance, dim_index, sub_portfolio,
                                                    self.p_random))
            else:
                portfolio, violations, new_x, new_phenotype = (
                    build_from_array_and_repair(phenotype, x[i], self.instance, self.p_random))
            phenotypes[i] = new_phenotype
            x[i] = new_x
            fits[i] = -portfolio.value  # negate as pymoo expects minimization
            results[i] = portfolio.result
            if self.stop_event and self.stop_event.is_set():
                print("Solver stopped.")
                return None  # Indicate that the solver was stopped

            if self.pause_event:
                self.pause_event.wait()  # Wait if paused

        out["F"] = fits
        out["X"] = x
        out["result"] = np.stack(results)
        out["pheno"] = np.stack(phenotypes)


def build_from_array_and_repair(phenotype, x, instance, p_random):
    planning_window = instance.planning_window
    num_projects = phenotype.shape[0]
    portfolio = Portfolio(num_projects, instance.budget_window, planning_window, instance.discount_rate,
                          instance.capability_stream_budgets.shape[0])

    # ignore projects that are not scheduled or are scheduled outside the planning window
    # np.random.permutation(range(num_projects)) will deteriorate the performance due to the prerequisite constraint
    for i in range(num_projects):
        if 0 < phenotype[i] <= planning_window:
            if portfolio.feasibility_check(instance.projects[i], phenotype[i], instance):
                portfolio.add_to_portfolio(i, phenotype[i], instance.projects[i])
            else:
                if np.random.random() < p_random:
                    portfolio = add_random_feasible(portfolio, i, instance.projects[i], instance)
                    x[i] = portfolio.result[i]
                    phenotype[i] = portfolio.result[i]
                else:
                    x[i] = 0
                    phenotype[i] = 0
        else:
            x[i] = 0
            phenotype[i] = 0

    violations = 0
    return portfolio, violations, x, phenotype


def add_random_feasible(portfolio, index, project, instance):
    time = find_random_feasible(portfolio, project, instance)
    if time > 0:
        portfolio.add_to_portfolio(index, time, project)
    return portfolio


def find_random_feasible(portfolio, project, instance):
    time = np.arange(instance.planning_window) + 1
    np.random.shuffle(time)
    for t in time:
        if portfolio.feasibility_check(project, t, instance):
            return t
    return -1


def build_from_sub_array_and_repair(phenotype, x, instance, sub_index, sub_portfolio, p_random):
    planning_window = instance.planning_window
    num_projects = phenotype.shape[0]

    fixed_index = np.setdiff1d(np.arange(num_projects), sub_index)

    # schedule the fixed_index
    if sub_portfolio is None:
        sub_portfolio = Portfolio(num_projects, instance.budget_window, planning_window, instance.discount_rate,
                                  instance.capability_stream_budgets.shape[0])
        for ind in range(len(fixed_index)):
            i = fixed_index[ind]
            if 0 < phenotype[i] <= planning_window:
                sub_portfolio.add_to_portfolio(i, phenotype[i], instance.projects[i])

    portfolio = copy.deepcopy(sub_portfolio)

    # schedule the sub_index
    for ind in range(len(sub_index)):
        i = sub_index[ind]
        if 0 < phenotype[i] <= planning_window:
            if portfolio.feasibility_check(instance.projects[i], phenotype[i], instance):
                portfolio.add_to_portfolio(i, phenotype[i], instance.projects[i])
            else:
                if np.random.random() < p_random:
                    portfolio = add_random_feasible(portfolio, i, instance.projects[i], instance)
                    x[i] = portfolio.result[i]
                    phenotype[i] = portfolio.result[i]
                else:
                    x[i] = 0
                    phenotype[i] = 0
        else:
            x[i] = 0
            phenotype[i] = 0

    violations = portfolio.constraint_violations(instance)

    return portfolio, violations, x, phenotype


def build_from_sub_array(best, instance, sub_index):
    """
    build the portfolio based on a subset of feasible solution
    """
    planning_window = instance.planning_window
    num_projects = best.shape[0]
    sub_portfolio = Portfolio(num_projects, instance.budget_window, planning_window, instance.discount_rate,
                              instance.capability_stream_budgets.shape[0])

    fixed_index = np.setdiff1d(np.arange(num_projects), sub_index)
    for ind in range(len(fixed_index)):
        i = fixed_index[ind]
        if 0 < best[i] <= planning_window:
            sub_portfolio.add_to_portfolio(i, best[i], instance.projects[i])
        # else:
        #     print("Warning: the best if not feasible")

    sub_violations = sub_portfolio.constraint_violations(instance)

    return sub_portfolio, sub_violations


def portfolio_local_search(instance, p_random, portfolio):
    planning_window = instance.planning_window
    phenotype = portfolio.result
    num_projects = phenotype.shape[0]

    # heuristics for projects that are not scheduled or are scheduled outside the planning window
    for i in range(num_projects):
        if phenotype[i] == 0 or phenotype[i] > planning_window:
            if np.random.random() < p_random:
                portfolio = add_random_feasible(portfolio, i, instance.projects[i], instance)

    return portfolio


def portfolio_mutate(instance, portfolio):
    phenotype = portfolio.result
    num_projects = phenotype.shape[0]
    num_mutants = 20
    new_portfolio = copy.deepcopy(portfolio)

    # heuristics for projects selected randomly
    permutation = np.random.choice(range(num_projects), num_mutants)
    # remove from portfolio
    for i in permutation:
        if new_portfolio.result[i] != 0:
            new_portfolio.remove_from_portfolio(i, instance.projects[i])

    for i in permutation:
        new_portfolio.add_earliest_feasible(i, instance.projects[i], instance)

    return new_portfolio


def portfolio_mutate_sequentially(instance, portfolio):
    phenotype = portfolio.result
    num_projects = phenotype.shape[0]
    num_mutants = 20
    new_portfolio = copy.deepcopy(portfolio)

    # heuristics for projects selected randomly
    permutation = np.random.choice(range(num_projects), num_mutants)
    # remove from portfolio
    for i in permutation:
        if new_portfolio.result[i] != 0:
            new_portfolio.remove_from_portfolio(i, instance.projects[i])
        new_portfolio.add_earliest_feasible(i, instance.projects[i], instance)

    return new_portfolio


def heuristic_local(pop, instance, random_seed, num_heuristic):
    new_individual = copy.deepcopy(pop[0])
    solution = pop[0].get("X")
    portfolio, violations = build_from_array(solution, instance)
    for i in range(num_heuristic):
        new_portfolio = portfolio_mutate_sequentially(instance, portfolio)
        if new_portfolio.value > portfolio.value:
            portfolio = copy.deepcopy(new_portfolio)

    new_individual.set("F", np.array([-portfolio.value]))
    new_individual.set("X", portfolio.result)
    new_individual.set("feasible", np.array([True]))
    new_individual.set("result", portfolio.result)
    new_individual.set("pheno", portfolio.result)

    return new_individual


def build_sub_instance_from_sub_solution(instance, sub_portfolio, sub_index):
    sub_instance = copy.deepcopy(instance)
    # extract the parameters from sub_portfolio
    sub_instance.budget -= sub_portfolio.cost
    sub_instance.capability_stream_budgets -= sub_portfolio.capability_stream_costs
    sub_instance.ongoing_budget -= sub_portfolio.ongoing_costs
    sub_instance.initiation_budget[:len(sub_portfolio.start_costs)] -= sub_portfolio.start_costs
    sub_instance.num_projects = len(sub_index)
    # obtain the projects in the sub_instance
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