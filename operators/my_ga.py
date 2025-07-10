from collections import namedtuple
from math import sqrt

import numpy as np
import time

# =========================================================================================================
# Implementation
# =========================================================================================================
from operators.iteration_data import IterationData
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.algorithm import filter_optimum
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.fitness_survival import FitnessSurvival
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_and_fitness
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display import SingleObjectiveDisplay


class MyGA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
                 crossover=SimulatedBinaryCrossover(prob=0.9, eta=3),
                 mutation=PolynomialMutation(prob=None, eta=5),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=SingleObjectiveDisplay(resolution=100),
                 verbose=False,
                 num_elites=3,
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=FitnessSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         verbose=verbose,
                         **kwargs)

        #self.verbose=True # TODO: this isn't seeming to wo
        # rk if set anywhere else..
        #self.func_display_attrs = disp_single_objective
        self.default_termination = MaximumGenerationTermination(1000)
        self.num_elites = num_elites

        # reduce the number of offspring by the number of elites to ensure population size stays the same
        self.n_offsprings -= self.num_elites
        #self.history = dict()

    def _each_iteration(self, *args, **kwargs):
        # display the output if defined by the algorithm
        if self.verbose and self.display is not None:
            self.display.do(self.problem, self.evaluator, self, pf=self.pf)

            # if a callback function is provided it is called after each iteration
            if self.callback is not None:
                # use the callback here without having the function itself
                self.callback(self)

        if self.save_history:
            hist, _callback = self.history, self.callback
            self.history, self.callback = None, None

            #obj = copy.deepcopy(self)
            fitnesses = -self.pop.get("F")
            best_sol = filter_optimum(self.pop, least_infeasible=True)[0]

            stats = basic_stats(fitnesses)

            obj = IterationData(self.n_gen, -best_sol.F[0], stats.mean, stats.std, stats.min, stats.max,
                                np.copy(best_sol.X))
            self.history = hist
            self.callback = _callback
            self.history.append(obj)

        if self.save_opt_intervals:
            exec_time = time.time() - self.start_time
            #print(exec_time)
            #if round(exec_time % 60) <= 1:
            self.time_history.append([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])

    def _next(self):

        # do the mating using the current population
        self.off = self.mating.do(self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self)
        self.off.set("n_gen", self.n_gen)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        #self.pop = self.pop.merge(self.off)

        # create new population by getting the elites and merging with the offspring
        self.pop = self.survival.do(self.problem, self.pop, self.num_elites, algorithm=self)
        self.pop = self.pop.merge(self.off)


def basic_stats(arr):
    min_val = float("inf")
    max_val = float("-inf")

    sum_val = 0
    count = 0

    for x in arr:
        val = x[0]
        min_val = min(val, min_val)
        max_val = max(val, max_val)
        sum_val += val
        count += 1

    mean = sum_val / count
    tmp = sum([pow(x[0] - mean, 2) for x in arr])
    std = sqrt(tmp / (count - 1))

    Stats = namedtuple("Stats", 'mean std min max')

    return Stats(mean, std, min_val, max_val)
