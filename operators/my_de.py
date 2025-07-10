# Jing Liu

import numpy as np
import time

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.crossover.differential_evolution_crossover import DifferentialEvolutionCrossover
from pymoo.operators.crossover.exponential_crossover import ExponentialCrossover
# from pymoo.operators.repair.bounds_back_repair import BoundsBackRepair
from pymoo.operators.repair.bounce_back_repair import BounceBackOutOfBoundsRepair
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.operators.selection.random_selection import RandomSelection
from pymoo.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_and_fitness
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.misc import parameter_less
from pymoo.operators.fitness_survival import FitnessSurvival
from pymoo.model.algorithm import filter_optimum
from operators.my_ga import basic_stats
from operators.iteration_data import IterationData
# =========================================================================================================
# Implementation
# =========================================================================================================


class MyDE(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
                 variant="DE/rand/1/bin",
                 CR=0.5,
                 F=0.3,
                 dither="vector",
                 jitter=False,
                 display=SingleObjectiveDisplay(resolution=100),
                 num_elites=3,
                 eliminate_duplicates=True,
                 verbose=False,
                 **kwargs
                 ):

        """

        Parameters
        ----------

        pop_size : {pop_size}

        sampling : {sampling}

        variant : {{DE/(rand|best)/1/(bin/exp)}}
         The different variants of DE to be used. DE/x/y/z where x how to select individuals to be pertubed,
         y the number of difference vector to be used and z the crossover type. One of the most common variant
         is DE/rand/1/bin.

        F : float
         The weight to be used during the crossover.

        CR : float
         The probability the individual exchanges variable values from the donor vector.

        dither : {{'no', 'scalar', 'vector'}}
         One strategy to introduce adaptive weights (F) during one run. The option allows
         the same dither to be used in one iteration ('scalar') or a different one for
         each individual ('vector).

        jitter : bool
         Another strategy for adaptive weights (F). Here, only a very small value is added or
         subtracted to the weight used for the crossover for each individual.


        """

        _, self.var_selection, self.var_n, self.var_mutation, = variant.split("/")

        if self.var_mutation == "exp":
            mutation = ExponentialCrossover(CR)
        elif self.var_mutation == "bin":
            mutation = BiasedCrossover(CR)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
                         crossover=DifferentialEvolutionCrossover(weight=F, dither=dither, jitter=jitter),
                         mutation=mutation,
                         survival=FitnessSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         display=display,
                         verbose=verbose,
                         **kwargs)

        self.default_termination = MaximumGenerationTermination(500)  # SingleObjectiveDefaultTermination()


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
            self.time_history.append ([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])



    def _next(self):
        selection, crossover, mutation = self.mating.selection, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        # get the vectors from the population
        F, CV, feasible = pop.get("F", "CV", "feasible")
        F = parameter_less(F, CV)

        # create offsprings and add it to the data of the algorithm
        if self.var_selection == "rand":
            P = selection.do(pop, self.pop_size, crossover.n_parents)

        elif self.var_selection == "best":
            best = np.argmin(F[:, 0])
            P = selection.do(pop, self.pop_size, crossover.n_parents - 1)
            P = np.column_stack([np.full(len(pop), best), P])

        elif self.var_selection == "rand+best":
            best = np.argmin(F[:, 0])
            P = selection.do(pop, self.pop_size, crossover.n_parents)
            use_best = np.random.random(len(pop)) < 0.3
            P[use_best, 0] = best

        else:
            raise Exception("Unknown selection: %s" % self.var_selection)

        # do the first crossover which is the actual DE operation
        self.off = crossover.do(self.problem, pop, P, algorithm=self)

        # then do the mutation (which is actually )
        _pop = self.off.new().merge(self.pop).merge(self.off)
        _P = np.column_stack([np.arange(len(pop)), np.arange(len(pop)) + len(pop)])
        self.off = mutation.do(self.problem, _pop, _P, algorithm=self)[:len(self.pop)]

        # bounds back if something is out of bounds
        # self.off = BounceBackOutOfBoundsRepair().do(self.problem, self.off) # seed int

        # evaluate the results
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        _F, _CV, _feasible = self.off.get("F", "CV", "feasible")
        _F = parameter_less(_F, _CV)

        # find the individuals which are indeed better
        is_better = np.where((_F <= F)[:, 0])[0]

        # replace the individuals in the population
        pop[is_better] = self.off[is_better]

        # store the population in the algorithm object
        self.pop = pop
