# Jing Liu

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
from operators.my_ga import basic_stats
import pathlib
import os
# from executor import get_gurobi_start_from_ea_solutions, run_gurobi_with_seeds
from solvers.gurobi_solver import get_portfolio_from_gurobi_solution, GurobiSolver

from pymoo.model.individual import Individual
from pymoo.model.population import Population
from problem.portfolio import portfolio_from_pickle
from solvers.gurobi_solver_local import GurobiSolverLocal
from operators.gurobi_operators import local_search_gurobi, run_gurobi_for_decomposed_problem
from pymoo.util.misc import parameter_less


class MyHybridGA(GeneticAlgorithm):
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
        self.local_search_frequency = 200
        self.gurobi_decomposed = gurobi_decomposed
        self.grouping = grouping

        # reduce the number of offspring by the number of elites to ensure population size stays the same
        self.n_offsprings -= self.num_elites
        self.time_local_search = 60
        self.mip_gap = 100
        self.flag = 0
        self.gurobi_time_limit = gurobi_time_limit
        self.output_dir = output_dir

    def _each_iteration(self, *args, **kwargs):
        # display the output if defined by the algorithm
        if self.verbose and self.display is not None:
            self.display.do(self.problem, self.evaluator, self, pf=self.pf)

            # if a callback function is provided it is called after each iteration
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
            #print(exec_time)
            #if round(exec_time % 60) <= 1:
            self.time_history.append ([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])

    def next(self):

        if self.problem is None:
            raise Exception("You have to call the `setup(problem)` method first before calling next().")

        # call next of the implementation of the algorithm
        if not self.is_initialized:
            self.initialize()
            self.is_initialized = True
        else:
            self._next()
            self.n_gen += 1

        # set the optimum - only done if the algorithm did not do it yet
        self._set_optimum()

        # set whether the algorithm is terminated or not
        self.has_terminated = self.flag #(not self.termination.do_continue(self)) or self.mip_gap < 0.002

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
                num_groups = len(self.grouping)
                pop = self.pop
                X, F, CV, feasible = pop.get("X", "F", "CV", "feasible")
                F = parameter_less(F, CV)
                best = X[np.argmin(F[:, 0]), :]
                # decomposition-based Gurobi
                for i in range(num_groups):
                    # get the vectors from the population
                    dim_index = self.grouping[i]
                    best, best_fitness = run_gurobi_for_decomposed_problem(self.problem.instance, dim_index,
                                                                                     best, self.gurobi_time_limit,
                                                                                     self.output_dir, self.n_gen)
                    print(f"Optimised the {i}th group and obtained the best value {best_fitness}")
                    self.n_gen += 1
                    exec_time = time.time() - self.start_time
                    self.time_history.append([self.n_gen, self.evaluator.n_eval, best_fitness, exec_time])
                gurobi_pop = Population(0, individual=Individual())
                gurobi_pop = gurobi_pop.new("X", np.array([best]), "F", np.array([[-best_fitness]]), "CV", np.array([[0]]),
                                            "feasible", np.array([[True]]))
                self.pop = self.pop.merge(gurobi_pop)
                self.pop = self.survival.do(self.problem, self.pop, self.pop_size, algorithm=self)
                self.flag = 1
        else:
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
            # self.pop = self.pop.merge(self.off)

            # create new population by getting the elites and merging with the offspring
            self.pop = self.survival.do(self.problem, self.pop, self.num_elites, algorithm=self)
            self.pop = self.pop.merge(self.off)
