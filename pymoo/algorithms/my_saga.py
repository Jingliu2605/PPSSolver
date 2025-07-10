from collections import namedtuple
from math import sqrt

import numpy as np
import time
from pymoo.util.misc import parameter_less
import copy
import math
import os
import pickle
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
from problem.test_portfolio_with_repair import heuristic_local
from problem.test_portfolio_with_repair import portfolio_local_search
from problem.portfolio import build_from_array
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from operators.gurobi_operators import local_search_gurobi, run_gurobi_for_decomposed_problem, run_gurobi_for_decomposed_problem_multi_solutions


class MySAGA(GeneticAlgorithm):

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
                 output_dir=None,
                 gurobi_time_limit=60,
                 gurobi_decomposed=False,
                 grouping=None,
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

        # self.verbose=True # TODO: this isn't seeming to wo
        # rk if set anywhere else..
        # self.func_display_attrs = disp_single_objective
        self.default_termination = MaximumGenerationTermination(1000)
        self.num_elites = num_elites
        self.local_search_frequency = 200
        self.gurobi_decomposed = gurobi_decomposed
        self.grouping = grouping

        # reduce the number of offspring by the number of elites to ensure population size stays the same
        self.n_offsprings -= self.num_elites
        self.averageFitness = None
        self.pastAverageFitness = None
        self.mutation_rate_adjustment_trigger = 0.08
        self.cross_prob = self.mating.crossover.prob
        self.mutate_prob = self.mating.mutation.prob

        self.time_local_search = 60
        self.mip_gap = 100
        self.flag = 0
        self.gurobi_time_limit = gurobi_time_limit
        self.output_dir = output_dir
        self.initial_population = None

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

            # obj = copy.deepcopy(self)
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
            self.time_history.append([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])

    def adaptive_mating(self, pop, n_offsprings):
        # self.off = self.mating.do(self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self)

        # the population object to be used
        off = pop.new()

        # infill counter - counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0

        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self._adaptive_mating(pop, n_remaining)

            if self.eliminate_duplicates is not None:
                _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                # IMPORTANT: Interestingly, this makes a difference in performance
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_infills += 1

            # if no new offsprings can be generated within a pre-specified number of generations
            if n_infills > 100:
                break

        return off

    def _adaptive_mating(self, pop, n_remaining):
        selection, crossover, mutation = self.mating.selection, self.mating.crossover, self.mating.mutation

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_select = math.ceil(n_remaining / crossover.n_offsprings)

        # select the parents for the mating - just an index array
        parents = selection.do(pop, n_select, crossover.n_parents)
        # do the crossover using the parents index and the population - additional data provided if necessary

        self.adapt_cross(pop, parents)
        _off = crossover.do(self.problem, pop, parents, algorithm=self)
        # do the mutation on the offsprings created through crossover
        _off = mutation.do(self.problem, _off, algorithm=self)

        return _off

    def adapt_cross(self, pop, parents):
        F, CV, feasible = pop.get("F", "CV", "feasible")
        F = parameter_less(F, CV)
        averageFitness = np.mean(F)
        parentsFitness = np.min(F[parents], axis=1)
        parentsFitness = [parentsFitness[i][0] for i in range(len(parentsFitness))]
        parentsFitness = np.array(parentsFitness)
        cross_prob = np.zeros(shape=len(parents), dtype=np.double)
        mutate_prob = np.zeros(shape=len(parents), dtype=np.double)
        do_adapt = parentsFitness>averageFitness

        cross_prob[~do_adapt] = self.cross_prob
        mutate_prob[~do_adapt] = self.mutate_prob
        if any(do_adapt):
            cross_prob[do_adapt] = (self.cross_prob -
                                    (self.cross_prob-0.94)/(1+np.exp(self.pop_size*(averageFitness-parentsFitness[do_adapt])/(averageFitness-self.opt.get("F")[0][0]))))

            mutate_prob[do_adapt] = (self.mutate_prob -
                                    (self.mutate_prob-0.4)/(1+np.exp(self.pop_size*(averageFitness-parentsFitness[do_adapt])/(averageFitness-self.opt.get("F")[0][0]))))
        self.mating.crossover.prob = cross_prob
        self.mating.mutation.prob = np.hstack((mutate_prob, mutate_prob))

    def next(self):

        if self.problem is None:
            raise Exception("You have to call the `setup(problem)` method first before calling next().")

        # call next of the implementation of the algorithm
        if not self.is_initialized:
            self.initialize()
            self.initial_population = self.pop
            self.is_initialized = True
        else:
            self._next()
            self.n_gen += 1

        # set the optimum - only done if the algorithm did not do it yet
        self._set_optimum()

        # set whether the algorithm is terminated or not
        self.has_terminated = (self.flag == 4) #(not self.termination.do_continue(self)) or self.mip_gap < 0.002  #self.flag

        # if the algorithm has terminated call the finalize method
        if self.has_terminated:
            self.finalize()

        # do what needs to be done each generation
        self._each_iteration()

    def _next(self):

        # update the cross rate and mutation rate
        # self.adapt()

        # do the mating using the current population
        # self.off = self.mating.do(self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self)

        self.off = self.adaptive_mating(self.pop, self.n_offsprings)
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
        # self.pop = self.pop.merge(self.off)

        # create new population by getting the elites and merging with the offspring
        self.pop = self.survival.do(self.problem, self.pop, self.num_elites, algorithm=self)

        self.pop = self.pop.merge(self.off)

        # _local = heuristic_local(self.pop, self.problem.instance, self.seed, 10)
        # self.pop = self.pop.merge(_local)
        # self.pop[0] = _local


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

