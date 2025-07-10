from multiprocessing.spawn import freeze_support

import numpy as np
import skopt

from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


from executor import run_parallel, run_de
from instance_parameters import InstanceParameters
from problem.enums import Optimizer, SchedulingOrder, ValueFunction, SpreadDistribution
from problem.portfolio_selection_instance import generate_instance

runs = 5
func_evals = 50000

dim1 = Integer(name='pop_size', low=10, high=500)
dim2 = Real(name='F', low=0.0, high=2.0)
dim3 = Real(name='CR', low=0.0, high=1.0)
dim4 = Categorical(name='crossover', categories=['bin', 'exp'])
dim5 = Categorical(name='selection', categories=['rand', 'best'])
dim6 = Categorical(name='dither', categories=['no', 'scalar', 'vector'])
dim7 = Categorical(name='jitter', categories=[True, False])
dimensions = [dim1, dim2, dim3, dim4, dim5, dim6, dim7]

#instances = get_instances_from_directory(r"D:\Documents\UNSW\Instances\Comparison")

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
def _objective_parallel(pop_size, F, CR, crossover, selection, dither, jitter):
    results = run_parallel(Optimizer.DE, runs, pop_size, ("n_eval", func_evals), instance, "", "",
                           SchedulingOrder.EARLIEST, F=F, CR=CR, crossover=crossover, selection=selection,
                           dither=dither, jitter=jitter)
    fits = np.fromiter((-r.F[0] for r in results), np.double, count=results.shape[0])
    return -np.mean(fits)


# @use_named_args(dimensions=dimensions)
# def _objective_sequential(pop_size, F, CR):
#     fits = np.zeros(runs)
#     for i in range(runs):
#         res = run_de(("n_eval", func_evals), pop_size, instance, i,
#                      SchedulingOrder.EARLIEST, F=F, CR=CR)  # TODO: random seed is just i currently
#         fits[i] = -res.F[0]
#
#     return -np.mean(fits)


def main():
    initial_points = [[46, 0.09092946204241105, 0.31850628944174136, 'bin', 'rand', 'scalar', True]]  # best found by previous Bayesian run

    checkpoint_callback = skopt.callbacks.CheckpointSaver("./output/de_opt_check_250.pkl")
    result = gp_minimize(func=_objective_parallel,
                         dimensions=dimensions,
                         acq_func="gp_hedge",
                         x0=initial_points,
                         n_calls=50,
                         n_random_starts=10,
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
    fig.savefig("./output/de_convergence_250.pdf")

    axes = plot_objective(result, n_samples=100)
    fig = axes.flatten()[0].figure
    fig.show()
    fig.savefig("./output/de_objective_250.pdf")


if __name__ == '__main__':
    freeze_support()
    main()
