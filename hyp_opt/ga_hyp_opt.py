import os
import pathlib
from multiprocessing.spawn import freeze_support

import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from executor import run_parallel, run_deterministic_solvers
from operators.feasibility_budget_repair import FeasibilityBudgetRepair
from problem.enums import Optimizer, SchedulingOrder
from problem.portfolio_selection_instance import get_instances_from_directory
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.operators.mutation.swap_mutation import SwapMutation
from solvers.random_earliest_solver import RandomEarliestSolver
from solvers.random_latest_solver import RandomLatestSolver

runs = 5
func_evals = 50000
max_pop = 1000
random_seed = None

dim1 = Integer(name='pop_size', low=10, high=max_pop)
dim2 = Categorical(name='crossover', categories=['HUX', 'SBX', 'UX'])  # , 'PSX'])
dim3 = Real(name='crossover_rate', low=0.0, high=1.0)
dim4 = Categorical(name='mutation',
                   categories=['swap', 'scramble'])  # , 'polynomial'])  # removed PMX as it is too slow
dim5 = Real(name='mutation_rate', low=0.0, high=1.0)
dimensions = [dim1, dim3, dim5]

base_dir = r"/home/kyle/Documents/UNSW/"

instances = get_instances_from_directory(os.path.join(base_dir, "Instances", "Comparison", "Unconstrained_Revised"))

# for ins in instances:
#    ins.write_to_file(os.path.join(r'D:\Documents\UNSW\Instances\Comparison\Unconstrained_Revised', f"instance_{ins.identifier}.dat"))

instance = instances[0]
# seeds = np.ndarray((1000, len(instance.projects)), dtype=int)
# earliest_seeder = RandomEarliestSolver(instance.projects, instance.budget, instance.planning_window, 0)
# latest_seeder = RandomLatestSolver(instance.projects, instance.budget, instance.planning_window, 0)
# seeds[:500] = earliest_seeder.repeat(500, display=False)
# seeds[500:] = latest_seeder.repeat(500, display=False)

num_seeds = max_pop
seeds = np.ndarray((num_seeds, len(instance.projects)), dtype=int)
num_deterministic_seeds = 7
seeds[:num_deterministic_seeds] = run_deterministic_solvers(instance, False, random_seed)
# exact_solver = GurobiSolver(instance, 10)
# exact_portfolio, _, model = exact_solver.solve(log_to_console=False)
# seeds[0] = exact_portfolio.result
# print(exact_portfolio.result)

# portfolio, violations = build_from_array2(exact_portfolio.result, instance.projects, instance.budget,
#                                          instance.planning_window)
# print(portfolio.result)
# print(f"Exact Fitness: {portfolio.value}")
# print(f"Exact Gap: {model.MIPGap:.2%}")
# print(violations)

# TODO: implement seeder as mutation..

earliest_seeder = RandomEarliestSolver(instance.projects, instance.budget, instance.planning_window, random_seed)
latest_seeder = RandomLatestSolver(instance.projects, instance.budget, instance.planning_window, random_seed)
remaining = num_seeds - num_deterministic_seeds
earliest_count = remaining // 2
seeds[num_deterministic_seeds:earliest_count + num_deterministic_seeds] = earliest_seeder.repeat(earliest_count,
                                                                                                 display=False)
latest_count = remaining - earliest_count
seeds[earliest_count + num_deterministic_seeds:] = latest_seeder.repeat(latest_count, display=False)

repair = FeasibilityBudgetRepair(prob_repair=0.25)


@use_named_args(dimensions=dimensions)
def _objective_parallel(pop_size, crossover_rate, mutation_rate):
    print(f"Pop: {pop_size}, CR: {crossover_rate}, MR: {mutation_rate}")
    # if crossover == 'HUX':
    #     crossover = HalfUniformCrossover(prob=crossover_rate)
    # elif crossover == 'SBX':
    #     crossover = IntegerFromFloatCrossover(clazz=SimulatedBinaryCrossover, prob=crossover_rate, eta=30)
    # elif crossover == 'UX':
    #     crossover = UniformCrossover(prob=crossover_rate)
    # elif crossover == 'PSX':
    #     crossover = PortfolioShuffledCrossover(prob=crossover_rate, greedy=False)
    #
    # if mutation == 'swap':
    #     mutation = SwapMutation(prob=mutation_rate)
    # elif mutation == 'scramble':
    #     mutation = ScrambleMutation(prob=mutation_rate)
    # elif mutation == 'polynomial':
    #     mutation = IntegerFromFloatMutation(clazz=PolynomialMutation, prob=mutation_rate, eta=20)
    # elif mutation == 'insertion':
    #     mutation = InsertionMutation(prob=mutation_rate)

    crossover = HalfUniformCrossover(prob=crossover_rate)
    mutation = SwapMutation(prob=mutation_rate)

    results = run_parallel(Optimizer.GA, runs, pop_size, ("n_eval", func_evals), instances[0], "", "",
                           SchedulingOrder.EARLIEST, crossover=crossover, crossover_rate=crossover_rate,
                           mutation=mutation, mutation_rate=mutation_rate, repair=repair, seeds=seeds[:pop_size])
    fits = np.fromiter((-r.F[0] for r in results), np.double, count=results.shape[0])
    return -np.mean(fits)


def main():
    output_dir = os.path.join(base_dir, "Output", "HypOpt", "GA")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = skopt.callbacks.CheckpointSaver(os.path.join(output_dir, "ga_opt_check_250.pkl"))
    result = gp_minimize(func=_objective_parallel,
                         dimensions=dimensions,
                         acq_func="gp_hedge",
                         n_calls=250,
                         n_random_starts=50,
                         verbose=True,
                         callback=[checkpoint_callback])

    # result = forest_minimize(func=_objective_parallel,
    #                          dimensions=dimensions,
    #                          n_calls=10,
    #                          base_estimator="ET",
    #                          verbose=True)
    print("Best fitness:", -result.fun)
    print("Best parameters:", result.x)

    axes = plot_convergence(result)
    fig = axes.figure
    fig.show()
    fig.savefig(os.path.join(output_dir, "ga_convergence_250.pdf"))

    axes = plot_objective(result, n_samples=100)
    fig = axes.flatten()[0].figure
    fig.show()
    fig.savefig(os.path.join(output_dir, "ga_objective_250.pdf"))


if __name__ == '__main__':
    freeze_support()
    main()
