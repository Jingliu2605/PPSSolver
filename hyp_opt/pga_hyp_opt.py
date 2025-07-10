from multiprocessing.spawn import freeze_support

import numpy as np
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from executor import run_parallel
from problem.enums import Optimizer, SchedulingOrder
from problem.portfolio_selection_instance import get_instances_from_directory
from pymoo.operators.crossover.ordered_crossover import OrderedCrossover
from pymoo.operators.crossover.partially_mapped_crossover import PartiallyMappedCrossover
from pymoo.operators.crossover.similar_block_order_crossover import SimilarBlockOrderCrossover
from pymoo.operators.crossover.similar_job_order_crossover import SimilarJobOrderCrossover

runs = 5
func_evals = 50000

dim1 = Integer(name='pop_size', low=10, high=1000)
dim2 = Categorical(name='crossover', categories=['OX', 'SJOX', 'SBOX'])  # removed PMX as it is too slow
dim3 = Real(name='crossover_rate', low=0.0, high=1.0)
dim4 = Real(name='mutation_rate', low=0.0, high=1.0)
dimensions = [dim1, dim2, dim3, dim4]

instances = get_instances_from_directory(r"D:\Documents\UNSW\Instances\Comparison")


@use_named_args(dimensions=dimensions)
def _objective_parallel(pop_size, crossover, crossover_rate, mutation_rate):
    if crossover == 'OX':
        crossover = OrderedCrossover(prob=crossover_rate)
    elif crossover == 'PMX':
        crossover = PartiallyMappedCrossover(prob=crossover_rate)
    elif crossover == 'SJOX':
        crossover = SimilarJobOrderCrossover(prob=crossover_rate)
    elif crossover == 'SBOX':
        crossover = SimilarBlockOrderCrossover(prob=crossover_rate)

    results = run_parallel(Optimizer.PermGA, runs, pop_size, ("n_eval", func_evals), instances[0], "", "",
                           SchedulingOrder.EARLIEST,
                           crossover=crossover, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
    fits = np.fromiter((-r.F[0] for r in results), np.double, count=results.shape[0])
    return -np.mean(fits)


def main():
    initial_points = [[250, 'OX', 0.9, 0.1],  # the default parameter settings
                      [327, 'OX', 1.0, 0.5433122133079076]]  # best found by Bayesian run

    checkpoint_callback = skopt.callbacks.CheckpointSaver("output/pga_opt_check_250.pkl")
    result = gp_minimize(func=_objective_parallel,
                         dimensions=dimensions,
                         acq_func="gp_hedge",
                         x0=initial_points,
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
    fig.savefig("output/pga_convergence_250.pdf")

    axes = plot_objective(result, n_samples=100)
    fig = axes.flatten()[0].figure
    fig.show()
    fig.savefig("output/pga_objective_250.pdf")


if __name__ == '__main__':
    freeze_support()
    main()
