import math
import os
import time
from warnings import warn
import pandas as pd
import numpy as np
# from IPython.core.display_functions import display

from algorithm_factory import get_solver_by_name
from analysis import analyze_portfolio
from problem.enums import SchedulingOrder, Optimizer, Weighting
from operators.my_ga import MyGA
from operators.seeded_sampling import SeededSampling
from problem.portfolio import build_from_array, portfolio_from_pickle
from problem.portfolio_problem_with_repair import build_from_array_and_repair
from problem.portfolio_real_ordered_problem import PortfolioRealOrderedProblem, MyElementwiseDuplicateElimination
from pymoo.algorithms.so_brkga import BRKGA
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.operators.mutation.scramble_mutation import ScrambleMutation
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.optimize import minimize
from solvers.gurobi_solver import GurobiSolver

from solvers.roulette_solver import RouletteSolver
from util import display_results
from problem.portfolio import build_from_permutation

from problem.portfolio_problem_with_repair import PortfolioProblemWithRepair
from operators.my_de import MyDE
import pathlib

from pymoo.algorithms.hegcl import HEGCL



def read_gurobi_data(gurobi_summary, instance_name):
    gurobi_data = pd.read_csv(gurobi_summary)
    for i in range(len(gurobi_data)):
        if str(gurobi_data["Instance"][i]) == instance_name:
            bound = gurobi_data[" Bound"][i]
            exact_portfolio = gurobi_data[" Fitness"][i]
            break
    return bound, exact_portfolio


def run_optimizer(optimizer, pop_size, instance, termination, i, runs, output_dir,
                  scheduling_order=SchedulingOrder.EARLIEST,
                  display_each_run=True, analyze_output=False,
                  **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_instance = output_dir
    kwargs.setdefault("pymoo_verbose", True)
    kwargs.setdefault("bound", None)
    kwargs.setdefault("output_dir", output_dir)
    kwargs.setdefault('seeded_time', 0)

    print(f"Running the {i}th job of {optimizer.value} with {pop_size} individuals")

    total_time = 0
    fitness = np.zeros(runs)
    bests = np.empty(runs, dtype=object)
    results = np.empty(runs, dtype=object)
    # TODO: implement algorithm-specific parameters in run_<opt> methods using kwargs

    if optimizer is Optimizer.DE:
        res = run_de(termination, pop_size, instance, i, scheduling_order,  **kwargs)
    elif optimizer is Optimizer.BRKGA:
        res = run_brkga(termination, pop_size, instance, i, scheduling_order, **kwargs)
    elif optimizer is Optimizer.GA:
        res = run_seeded_ga(termination, pop_size, instance, i, **kwargs)
    elif optimizer is Optimizer.AGA:
        res = run_aga(termination, pop_size, instance, i, **kwargs)
    else:
        warn("Undefined optimizer type")

    total_time += res.exec_time

    fitness[0] = -res.F[0]
    kwargs.setdefault('pheno', None)
    pheno = kwargs['pheno']
    bests[0] = _get_best_by_optimizer(res, optimizer, pheno)
    results[0] = res

    if display_each_run:
        # create an anonymous object with the fields expected by display_results
        ga_solution = type('', (object,), {"result": None, "value": -res.F[0]})()

        display_results(ga_solution, f"{optimizer.value} {scheduling_order.value} ({pop_size})",
                        instance.planning_window, res.exec_time)
        print("Constraint violation: %s" % res.CV[0])

    if analyze_output:
        best = _get_best_by_optimizer(res, optimizer, pheno)

        if (optimizer is optimizer.GA or optimizer is Optimizer.AGA):
            portfolio, _ = build_from_array(best, instance)
        else:
            portfolio = build_from_permutation(best, instance, scheduling_order)

        analyze_portfolio(portfolio, instance, scheduling_order.value, True, True, output_dir_instance,
                          yearly_project_analysis=False)

    # write best values every 60 second
    if res.time_history is not None:
        pathlib.Path(output_dir_instance).mkdir(parents=True, exist_ok=True)
        num_intervals = len(res.time_history)
        with open(os.path.join(output_dir_instance, f"convergence_data_run_{i + 1}.csv"), "w") as alg_file:
            alg_file.write("Generations, Evaluations, Fitness, Time\n")
            for j in range(num_intervals):
                alg_file.write(
                    f"{res.time_history[j][0]}, {res.time_history[j][1]},  {-res.time_history[j][2][0][0]}, {res.time_history[j][3]}\n")

    bound = kwargs["bound"]

    best_fit = np.max(fitness)
    print(f"Average Fitness: {np.mean(fitness):.3f}")
    print(f"Best Fitness: {best_fit:.3f}")
    if bound is not None:
        print(f"Best Error (Bound): {(1 - best_fit / bound) * 100:.1f}%")

    print(f"Worst Fitness: {np.min(fitness):.3f}")
    print(f"Average Time: {total_time / runs:.3E}s")

    return results


def run_deterministic_solvers(instance, display_output, random_seed):
    deterministic_solvers = ["cyclicprefsolver", "cyclichighvaluesolver", "cycliclowvaluesolver",
                             "cyclichighcostsolver", "cycliclowcostsolver", "prefearliestsolver", "preflatestsolver"]
    # , "prefearliestconstraintsolver"]
    seeds = np.ndarray((len(deterministic_solvers), len(instance.projects)), dtype=int)
    index = 0
    # no need to run these more than once as they are not randomized
    for alg in deterministic_solvers:
        solver = get_solver_by_name(alg, instance, random_seed)
        seeds[index] = run_heuristic(solver, display_output)
        index += 1

    return seeds


def run_heuristic(heuristic, display=True):
    start_time = time.perf_counter()
    solution = heuristic.solve()
    if display:
        display_results(solution, heuristic.name, heuristic.instance.planning_window, time.perf_counter() - start_time)
    return np.array(solution.result, dtype=int)  # TODO: return solution?


def run_brkga(termination, pop_size, instance, random_seed, scheduling_order, **kwargs):
    np.random.seed(random_seed)

    kwargs.setdefault('bias', 0.7)
    kwargs.setdefault('prop_elites', 0.25)
    kwargs.setdefault('prop_mutants', 0.1)

    bias = kwargs['bias']
    prop_elites = kwargs['prop_elites']
    prop_mutants = kwargs['prop_mutants']

    kwargs.setdefault('stop_event', None)
    kwargs.setdefault('pause_event', None)
    stop_event = kwargs['stop_event']
    pause_event = kwargs['pause_event']
    kwargs.setdefault('gui_output', None)
    gui_output = kwargs['gui_output']
    display = kwargs['display'] if 'display' in kwargs else None

    num_elites = math.floor(prop_elites * pop_size)
    num_mutants = math.floor(prop_mutants * pop_size)
    num_offspring = pop_size - num_elites - num_mutants

    method = BRKGA(n_elites=num_elites, n_offsprings=num_offspring, n_mutants=num_mutants, bias=bias,
                   display=display,
                   eliminate_duplicates=MyElementwiseDuplicateElimination(), return_least_infeasible=True)

    problem = PortfolioRealOrderedProblem(instance, scheduling_order, pause_event=pause_event, stop_event=stop_event)
    res = minimize(problem, method, termination=termination, seed=random_seed, save_history=False,
                   save_opt_intervals=True,
                   verbose=kwargs["pymoo_verbose"], gui_output=gui_output)

    return res


def run_de(termination, pop_size, instance, random_seed, scheduling_order, **kwargs):
    np.random.seed(random_seed)

    kwargs.setdefault('CR', 0.5)
    kwargs.setdefault('F', 0.3)
    kwargs.setdefault('dither', 'vector')
    kwargs.setdefault('selection', 'rand')
    kwargs.setdefault('crossover', 'bin')
    kwargs.setdefault('jitter', False)

    cr = kwargs['CR']
    f = kwargs['F']
    dither = kwargs['dither']
    selection = kwargs['selection']
    crossover = kwargs['crossover']
    jitter = kwargs['jitter']

    kwargs.setdefault('stop_event', None)
    kwargs.setdefault('pause_event', None)
    stop_event = kwargs['stop_event']
    pause_event = kwargs['pause_event']
    kwargs.setdefault('gui_output', None)
    gui_output = kwargs['gui_output']

    variant = f"DE/{selection}/1/{crossover}"
    display = kwargs['display'] if 'display' in kwargs else None

    method = MyDE(pop_size=pop_size,
                  sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
                  variant=variant,
                  CR=cr,
                  F=f,
                  display=display,
                  dither=dither,
                  jitter=jitter
                  )

    problem = PortfolioRealOrderedProblem(instance, scheduling_order, pause_event=pause_event, stop_event=stop_event)
    res = minimize(problem, method, termination=termination, seed=random_seed, save_history=False,
                   save_opt_intervals=True,
                   verbose=kwargs["pymoo_verbose"], gui_output=gui_output)

    return res


def run_seeded_ga(termination, pop_size, instance, random_seed, **kwargs):
    kwargs.setdefault('crossover_rate', 0.9)
    crossover_rate = kwargs['crossover_rate']
    kwargs.setdefault('crossover', HalfUniformCrossover(prob=crossover_rate))
    crossover = kwargs['crossover']

    kwargs.setdefault('mutation_rate', 0.1)
    mutation_rate = kwargs['mutation_rate']
    kwargs.setdefault('mutation', ScrambleMutation(prob=mutation_rate))
    mutation = kwargs['mutation']

    kwargs.setdefault('seeded_time', 0)
    time_limitation = kwargs['seeded_time']
    output_dir = kwargs['output_dir']

    kwargs.setdefault('stop_event', None)
    kwargs.setdefault('pause_event', None)
    kwargs.setdefault('gui_output', None)
    stop_event = kwargs['stop_event']
    pause_event = kwargs['pause_event']
    gui_output = kwargs['gui_output']

    display = kwargs['display'] if 'display' in kwargs else None

    n_pool_solutions = 10
    pool_search_mode = 1
    kwargs.setdefault('old_solution', None)
    seeds = generate_seeds(instance, pop_size, random_seed, time_limitation, n_pool_solutions, pool_search_mode,
                           output_dir, pre_solution=kwargs['old_solution'])

    method = MyGA(pop_size=pop_size, sampling=SeededSampling(seeds), crossover=crossover, mutation=mutation,
                  eliminate_duplicates=True, return_least_infeasible=True, display=display,
                  verbose=True)

    problem = PortfolioProblemWithRepair(instance, pause_event=pause_event, stop_event=stop_event)
    # problem = PortfolioSelectionProblem(instance)
    res = minimize(problem, method, termination=termination, seed=random_seed, save_history=False,
                   save_opt_intervals=True, verbose=kwargs["pymoo_verbose"], gui_output=gui_output)

    return res


def run_aga(termination, pop_size, instance, random_seed, **kwargs):
    kwargs.setdefault('crossover_rate', 0.9)
    crossover_rate = kwargs['crossover_rate']
    kwargs.setdefault('crossover', HalfUniformCrossover(prob=crossover_rate))
    crossover = kwargs['crossover']

    kwargs.setdefault('mutation_rate', 0.1)
    mutation_rate = kwargs['mutation_rate']
    kwargs.setdefault('mutation', ScrambleMutation(prob=mutation_rate))
    mutation = kwargs['mutation']

    kwargs.setdefault('old_solution', None)

    kwargs.setdefault('seeds', generate_seeds(instance, pop_size, random_seed, pre_solution=kwargs['old_solution']))
    seeds = kwargs['seeds']
    output_dir = kwargs['output_dir']

    kwargs.setdefault('group_size', None)
    group_size = kwargs['group_size']

    kwargs.setdefault('seeded_time', 0)
    t_limitation = kwargs['seeded_time']
    kwargs.setdefault('seeded_time', 0)

    kwargs.setdefault('stop_event', None)
    kwargs.setdefault('pause_event', None)
    stop_event = kwargs['stop_event']
    pause_event = kwargs['pause_event']
    kwargs.setdefault('gui_output', None)
    gui_output = kwargs['gui_output']
    display = kwargs['display'] if 'display' in kwargs else None

    method = HEGCL(pop_size=pop_size, sampling=SeededSampling(seeds), crossover=crossover, mutation=mutation,
                  eliminate_duplicates=True, return_least_infeasible=True, gurobi_decomposed=True,
                  verbose=True, output_dir=output_dir, gurobi_time_limit=t_limitation, display=display,
                  group_size=group_size)

    problem = PortfolioProblemWithRepair(instance, pause_event=pause_event, stop_event=stop_event)
    res = minimize(problem, method, termination=termination, seed=random_seed, save_history=False,
                   save_opt_intervals=True, verbose=kwargs["pymoo_verbose"], gui_output=gui_output)

    return res


def run_gurobi_4_seeds(instance, time_limitation, n_solutions, pool_search_mode, output_dir):
    # run gurobi solver
    head, tail = os.path.split(output_dir)
    gurobi_seed_dir = os.path.join(head, "Gurobi")
    pathlib.Path(gurobi_seed_dir).mkdir(parents=True, exist_ok=True)
    gurobi_seed_file = os.path.join(gurobi_seed_dir, f"Gurobi-portfolio-{time_limitation}.pkl")
    gurobi_pool_solutions_file = os.path.join(gurobi_seed_dir, f"Gurobi-pool-solutions-{time_limitation}.npy")
    log_file = os.path.join(gurobi_seed_dir, f"Gurobi_{instance.identifier}_{time_limitation}.log")
    if not os.path.exists(gurobi_seed_file) or not os.path.exists(gurobi_pool_solutions_file):
        exact_solver = GurobiSolver(instance, time_limit=time_limitation, error_threshold=0.002,
                                    num_solutions=n_solutions, pool_search_mode=pool_search_mode)
        exact_portfolio, status, model, exec_time, pool_solutions, suboptimal = exact_solver.solve(verbose=True,
                                                                                                   log_to_console=True,
                                                                                                   log_file=log_file)
        exact_portfolio.write_to_pickle(gurobi_seed_file)
        np.save(gurobi_pool_solutions_file, pool_solutions)
        print(f"Gurobi solution's fitness: {exact_portfolio.value:.3f}")
    else:
        exact_portfolio = portfolio_from_pickle(gurobi_seed_file)
        pool_solutions = np.load(gurobi_pool_solutions_file)
        pool_solutions = pool_solutions[0]
        print(f"Gurobi solution's fitness: {exact_portfolio.value:.3f}")
    return exact_portfolio, pool_solutions


def _get_best_by_optimizer(res, optimizer, pheno=None):
    if pheno is True:
        best = res.algorithm.opt[0].get("pheno")
    elif pheno is False:
        best = res.X
    elif pheno is None:
        if (optimizer is Optimizer.Gurobi_DE or optimizer is Optimizer.Gurobi_DE_S ):
            best = res.algorithm.opt[0].get("pheno")
        else:
            best = res.X
    return best


def generate_seeds(instance, num_seeds, random_seed, gurobi_time_limitation=0, n_pool_solutions=None,
                   pool_search_mode=None, output_dir=None, pre_solution=None):
    seeds = np.ndarray((num_seeds, len(instance.projects)), dtype=int)

    if gurobi_time_limitation != 0:
        _, pool_solutions = run_gurobi_4_seeds(instance, gurobi_time_limitation, n_pool_solutions,
                                               pool_search_mode, output_dir)
        num_real_pool_solutions = 1  # len(pool_solutions)
        seeds[:num_real_pool_solutions] = pool_solutions
    else:
        num_real_pool_solutions = 0

    # pre_solution and gurobi are not supposed to be used at the same time
    if pre_solution is not None:
        for i in range(10):
            portfolio, violation, x, phenotype = build_from_array_and_repair(pre_solution, pre_solution, instance, 0.5)
            seeds[i] = x
            num_real_pool_solutions += 1

    num_deterministic_seeds = 7
    seeds[num_real_pool_solutions:num_deterministic_seeds + num_real_pool_solutions] = run_deterministic_solvers(
        instance, False, random_seed)

    num_deterministic_seeds = num_deterministic_seeds + num_real_pool_solutions

    earliest_seeder = RouletteSolver(instance, random_seed, Weighting.VC_RATIO, SchedulingOrder.EARLIEST)

    latest_seeder = RouletteSolver(instance, random_seed, Weighting.VC_RATIO, SchedulingOrder.LATEST)

    remaining = num_seeds - num_deterministic_seeds
    earliest_count = remaining // 2
    seeds[num_deterministic_seeds:earliest_count + num_deterministic_seeds] = earliest_seeder.repeat(earliest_count,
                                                                                                     display=False)
    latest_count = remaining - earliest_count
    seeds[earliest_count + num_deterministic_seeds:] = latest_seeder.repeat(latest_count, display=False)

    return seeds


def report_heuristic(instance, optimizer, index, output_dir, runs,
                     results, start_time_rep=None, pheno=None, seeded_time=0):
    alg_dir = output_dir
    fitnesses = np.zeros(runs)
    generations = np.zeros(runs)
    evals = np.zeros(runs)
    times = np.zeros(runs)
    with (open(os.path.join(alg_dir, f"results_run_{index + 1}.csv"), "w") as alg_file):
        alg_file.write("Run, Fitness, Generations, Evaluations, Time\n")
        for j in range(runs):
            res = results[j]
            best = _get_best_by_optimizer(res, optimizer, pheno)
            if start_time_rep:  # start time representation
                portfolio, _ = build_from_array(best, instance)
            else:   # permutation representation
                portfolio = build_from_permutation(best, instance, SchedulingOrder.EARLIEST)

            portfolio.write_to_pickle(os.path.join(alg_dir, f"Portfolio_{index + 1}.pkl"))
            fitnesses[j] = portfolio.value
            generations[j] = res.algorithm.n_gen
            evals[j] = res.algorithm.evaluator.n_eval
            times[j] = res.exec_time

            alg_file.write(f"{j + 1}, {portfolio.value}, {res.algorithm.n_gen}, "
                           f"{res.algorithm.evaluator.n_eval}, {res.exec_time}\n")
    return fitnesses, times, generations, evals


