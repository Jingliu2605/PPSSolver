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
from operators.gurobi_operators import local_search_gurobi, run_gurobi_for_decomposed_problem, \
    run_gurobi_for_decomposed_problem_multi_solutions
from grouping import problem_dependent_random_grouping


class MyAGA(GeneticAlgorithm):

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
                 group_size=None,
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

        self.default_termination = MaximumGenerationTermination(1000)
        self.num_elites = num_elites
        self.local_search_frequency = 50
        self.gurobi_decomposed = gurobi_decomposed
        self.group_size = group_size

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
        do_adapt = parentsFitness > averageFitness

        cross_prob[~do_adapt] = self.cross_prob
        mutate_prob[~do_adapt] = self.mutate_prob
        if any(do_adapt):
            cross_prob[do_adapt] = (self.cross_prob -
                                    (self.cross_prob - 0.94) / (1 + np.exp(
                        self.pop_size * (averageFitness - parentsFitness[do_adapt]) / (
                                    averageFitness - self.opt.get("F")[0][0]))))

            mutate_prob[do_adapt] = (self.mutate_prob -
                                     (self.mutate_prob - 0.5) / (1 + np.exp(
                        self.pop_size * (averageFitness - parentsFitness[do_adapt]) / (
                                    averageFitness - self.opt.get("F")[0][0]))))
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
        self.has_terminated = (
                    self.flag == 2)  # (not self.termination.do_continue(self)) or self.mip_gap < 0.002  #self.flag

        # if the algorithm has terminated call the finalize method
        if self.has_terminated:
            self.finalize()

        # do what needs to be done each generation
        self._each_iteration()

    def _next(self):
        if self.n_gen % self.local_search_frequency == 0:
            if self.gurobi_decomposed is False:
                # run gurobi
                start_solutions, start_fitness, model = local_search_gurobi(self.pop[0], self.problem.instance,
                                                                            self.flag,
                                                                            self.output_dir, self.gurobi_time_limit)
                gurobi_pop = Population(0, individual=Individual())
                if self.gurobi_time_limit is not None:
                    start_fitness_tuple = [np.array([-start_fitness[i]]) for i in range(len(start_fitness))]
                    cv_value = [np.array([0]) for i in range(len(start_fitness))]
                    feasible_value = [np.array([True]) for i in range(len(start_fitness))]
                else:  # only one solution
                    start_solutions = [start_solutions]
                    start_fitness_tuple = [np.array([-start_fitness])]
                    cv_value = [np.array([0])]
                    feasible_value = [np.array([True])]

                gurobi_pop = gurobi_pop.new("X", start_solutions, "F", start_fitness_tuple, "CV", cv_value, "feasible",
                                            feasible_value)
                # self.pop = Population.merge(self.pop, gurobi_pop)
                self.pop = self.pop.merge(gurobi_pop)
                self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)
                self.flag = 1
                self.mip_gap = model.MIPGap
            else:
                # if self.flag != 0:
                size_groups = self.group_size + self.flag * 100
                grouping_results = problem_dependent_random_grouping(self.problem.instance, size_groups)
                if self.gurobi_time_limit is not None:
                    gurobi_time_limit = self.gurobi_time_limit + self.flag * 10
                else:
                    gurobi_time_limit = None
                # head, tail = os.path.split(self.output_dir)
                # file_name = os.path.join(head, f"grouping_{size_groups}.pkl")
                # self.grouping = pickle.load(open(file_name, "rb"))

                num_groups = len(grouping_results)
                pop = self.pop
                X, F, CV, feasible = pop.get("X", "F", "CV", "feasible")
                F = parameter_less(F, CV)
                best = X[np.argmin(F[:, 0]), :]
                new_x = []
                new_y = []
                new_cv = []
                new_feasible = []

                # decomposition-based Gurobi
                for i in range(num_groups):
                    # get the vectors from the population
                    dim_index = grouping_results[i]
                    new_best, new_best_fitness = run_gurobi_for_decomposed_problem_multi_solutions(
                        self.problem.instance, dim_index,
                        best, gurobi_time_limit,
                        self.output_dir, self.n_gen)
                    best = new_best[0]
                    current_best_fitness = new_best_fitness[0]

                    # start_fitness_tuple = [np.array([-new_best_fitness[0]])]
                    # cv_value = [np.array([0])]
                    # feasible_value = [np.array([True])]

                    new_x.extend([new_best[0]])
                    new_y.extend([np.array([-new_best_fitness[0]])])
                    new_cv.extend([np.array([0])])
                    new_feasible.extend([np.array([True])])

                    self.n_gen += 1
                    exec_time = time.time() - self.start_time
                    self.time_history.append(
                        [self.n_gen, self.evaluator.n_eval, np.array([[-current_best_fitness]]), exec_time])

                    print(f"Optimised the {i}th group and obtained the best value {current_best_fitness}")

                portfolio, violations = build_from_array(best, self.problem.instance)  # 4819640
                portfolio = portfolio_local_search(self.problem.instance, 0.5, portfolio)

                if portfolio.value > current_best_fitness:
                    best = portfolio.result
                    current_best_fitness = portfolio.value
                    new_x.extend([best])
                    new_y.extend([np.array([-current_best_fitness])])
                    new_cv.extend([np.array([0])])
                    new_feasible.extend([np.array([True])])

                print(f"local search found best value {current_best_fitness}")
                gurobi_pop = Population(0, individual=Individual())
                gurobi_pop = gurobi_pop.new("X", new_x, "F", new_y, "CV", new_cv, "feasible", new_feasible)

                # gurobi_pop = gurobi_pop.new("X", np.array([best]), "F", np.array([[-current_best_fitness]]), "CV",
                #                             np.array([[0]]), "feasible", np.array([[True]]))
                self.pop = self.initial_population.merge(gurobi_pop)
                # self.pop = self.pop.merge(gurobi_pop)
                self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)
                self.flag += 1
        else:
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

        # def adapt(self, pop):
        #     F, CV, feasible = pop.get("F", "CV", "feasible")
        #     F = parameter_less(F, CV)
        #     if self.averageFitness is None:
        #         self.averageFitness = np.mean(F)
        #         self.pastAverageFitness = copy.deepcopy(self.averageFitness)
        #     else:
        #         self.pastAverageFitness = copy.deepcopy(self.averageFitness)
        #         self.averageFitness = np.mean(F)
        #
        #     # Increase mutation rate for low performing generations and decrease for good performance
        #     if ((self.averageFitness - self.pastAverageFitness > 0) or (
        #             abs(self.averageFitness - self.pastAverageFitness) <= self.mutation_rate_adjustment_trigger)
        #             and not self.mating.mutation.prob >= 1 and not self.n_gen == 1):
        #         self.mating.mutation.prob += .05
        #     elif self.mating.mutation.prob > .10:
        #         self.mating.mutation.prob -= .05
        #     self.mating.mutation.prob = round(self.mating.mutation.prob, 2)


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
