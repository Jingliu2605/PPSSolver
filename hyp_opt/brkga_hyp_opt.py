from multiprocessing.spawn import freeze_support

import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from executor import run_parallel, run_brkga
from instance_parameters import InstanceParameters
from problem.enums import Optimizer, SchedulingOrder, ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import get_instances_from_directory, generate_instance

runs = 5
func_evals = 50000

dim1 = Integer(name='pop_size', low=10, high=500)
dim2 = Real(name='bias', low=0.05, high=0.95)
dim3 = Real(name='prop_elites', low=0.1, high=0.5)
dim4 = Real(name='prop_mutants', low=0.1, high=0.4)
dimensions = [dim1, dim2, dim3, dim4]

# instances = get_instances_from_directory(r"D:\Documents\UNSW\Instances\Comparison")

parameters = InstanceParameters(num_projects=1000,
                                planning_window=20,
                                base_budget=14439,  # value taken from 2020 FSP
                                budget_increase=1637,  # value taken from 2020 FSP
                                capability_stream_proportions=np.array([0.346, 0.296, 0.247, 0.074, 0.037]),
                                # from 2020 FSP
                                initiation_max_proportion=0.25,
                                ongoing_max_proportion=0.75,
                                prerequisite_tuples=[(2, 0.10)],
                                exclusion_tuples=[(2, 0.05), (3, 0.45)],
                                completion_constraint_chance=0,
                                value_func=ValueFunction.COST_DUR,
                                cost_distribution=SpreadDistribution.WEIBULL,
                                discount_rate=0.01
                                )

instance = generate_instance(parameters, 0, False)

@use_named_args(dimensions=dimensions)
def _objective_parallel(pop_size, bias, prop_elites, prop_mutants):
    num_instances = 1
    means = np.zeros(num_instances)
    for i in range(num_instances):
        results = run_parallel(Optimizer.BRKGA, runs, pop_size, ("n_eval", func_evals), instance, "", "",
                               SchedulingOrder.EARLIEST, bias=bias, prop_elites=prop_elites, prop_mutants=prop_mutants)
        # TODO: find easy way to get fitnesses
        fits = np.fromiter((-r.F[0] for r in results), np.double, count=results.shape[0])
        means[i] = np.mean(fits)
    return -np.mean(means)


@use_named_args(dimensions=dimensions)
def _objective_sequential(pop_size, bias, prop_elites, prop_mutants):
    fits = np.zeros(runs)
    for i in range(runs):
        res = run_brkga(("n_eval", func_evals), pop_size, instance, i,  # TODO: random seed is just i currently
                        SchedulingOrder.EARLIEST, bias=bias, prop_elites=prop_elites, prop_mutants=prop_mutants)
        fits[i] = -res.F[0]

    return -np.mean(fits)  # negate as we want to maximize


def main():
    # initial points to be investigated by the optimizer
    initial_points = [[279, 0.4136926099667668, 0.24305100370811983, 0.10479814950439764]]  # best found by Bayesian run

    checkpoint_callback = skopt.callbacks.CheckpointSaver("./output/brkga_opt_check_250.pkl.pkl")
    # result = forest_minimize(func=_objective_parallel,
    #                          dimensions=dimensions,
    #                          x0=initial_points,
    #                          n_calls=100,
    #                          base_estimator="ET",
    #                          verbose=True,
    #                          callback=[checkpoint_callback])
    #
    # print("ET Best fitness:", -result.fun)
    # print("ET Best parameters:", result.x)

    result = gp_minimize(func=_objective_parallel,
                         dimensions=dimensions,
                         acq_func="gp_hedge",
                         x0=initial_points,  # provide the defaults as initial point
                         n_calls=50,
                         n_random_starts=10,
                         verbose=True,
                         callback=[checkpoint_callback])

    print("GP Best fitness:", -result.fun)
    print("GP Best parameters:", result.x)

    axes = plot_convergence(result)
    fig = axes.figure
    fig.show()
    fig.savefig("./output/brkga_convergence.pdf")

    axes = plot_objective(result, n_samples=100)
    fig = axes.flatten()[0].figure
    fig.show()
    fig.savefig("./output/brkga_objective.pdf")


if __name__ == '__main__':
    freeze_support()
    main()
