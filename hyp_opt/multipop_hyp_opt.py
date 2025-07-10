from multiprocessing.spawn import freeze_support

import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Integer
from skopt.utils import use_named_args

from executor import run_parallel, run_brkga
from instance_parameters import InstanceParameters
from problem.enums import Optimizer, SchedulingOrder, ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import get_instances_from_directory, generate_instance

runs = 5
func_evals = 50000

dim1 = Integer(name='merge_frequency', low=1, high=100)
dim2 = Integer(name='n_transfer', low=1, high=25)

dimensions = [dim1, dim2]

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
def _objective_parallel(merge_frequency, n_transfer):
    num_instances = 1
    means = np.zeros(num_instances)
    for i in range(num_instances):
        results = run_parallel(Optimizer.DE_BRKGA, runs, 0, ("n_eval", func_evals), instance, "", "",
                               SchedulingOrder.EARLIEST, merge_frequency=merge_frequency, n_transfer=n_transfer)
        fits = np.fromiter((-r.F[0] for r in results), np.double, count=results.shape[0])
        means[i] = np.mean(fits)
    return -np.mean(means)


def main():
    # initial points to be investigated by the optimizer
    initial_points = [[50, 3], [25, 10]]

    checkpoint_callback = skopt.callbacks.CheckpointSaver("./output/de-brkga_opt_check_250.pkl")
    # result = forest_minimize(func=_objective_parallel,
    #                          dimensions=dimensions,
    #                          x0=initial_points,
    #                          n_calls=100,
    #                          base_estimator="ET",
    #                          verbose=True,
    #                          callback=[checkpoint_callback])
    #1
    # print("ET Best fitness:", -result.fun)
    # print("ET Best parameters:", result.x)

    result = gp_minimize(func=_objective_parallel,
                         dimensions=dimensions,
                         acq_func="gp_hedge",
                         n_calls=50,
                         n_random_starts=10,
                         # x0=initial_points,
                         verbose=True,
                         callback=[checkpoint_callback])

    print("GP Best fitness:", -result.fun)
    print("GP Best parameters:", result.x)

    axes = plot_convergence(result)
    fig = axes.figure
    fig.show()
    fig.savefig("./output/de-brkga_convergence.pdf")

    axes = plot_objective(result, n_samples=100)
    fig = axes.flatten()[0].figure
    fig.show()
    fig.savefig("./output/de-brkga_objective.pdf")


if __name__ == '__main__':
    freeze_support()
    main()
