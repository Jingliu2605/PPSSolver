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
from scipy.stats import cauchy
# =========================================================================================================
# Implementation
# =========================================================================================================


class JADE(GeneticAlgorithm):
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
        self.mu = 0.5  # mean of normal distribution for adaptation of crossover probabilities
        self.median = 0.5  # location of Cauchy distribution for adaptation of mutation factor
        self.p = 0.05  # level of greediness of the mutation strategy
        assert 0.0 <= self.p <= 1.0
        self.c = 0.1  # life span
        assert 0.0 <= self.c <= 1.0
        self.is_bound = True
        self.a = None


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
            self.time_history.append([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])

    def _initialize(self):
        # Todo: test
        GeneticAlgorithm._initialize(self)

        # # create the initial population
        # pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        # pop.set("n_gen", self.n_gen)
        #
        # # then evaluate using the objective function
        # self.evaluator.eval(self.problem, pop, algorithm=self)
        #
        # # that call is a dummy survival to set attributes that are necessary for the mating selection
        # if self.survival:
        #     pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
        #                            n_min_infeas_survive=self.min_infeas_pop_size)
        #
        # self.pop, self.off = pop, pop

        a = np.empty((0, self.problem.n_var)) # set of archived inferior solutions
        self.a = a

    def bound(self, x=None, xx=None):
        pop_size = len(x)
        if not self.is_bound:
            return x
        for k in range(pop_size):
            idx = np.array(x[k] < self.problem.xl)
            if idx.any():
                x[k][idx] = (self.problem.xl + xx[k])[idx]/2.0
            idx = np.array(x[k] > self.problem.xu)
            if idx.any():
                x[k][idx] = (self.problem.xu + xx[k])[idx]/2.0
        return x

    def _next(self):
        # retrieve the current population
        pop = self.pop
        a = self.a

        # get the vectors from the population
        y, CV, feasible = pop.get("F", "CV", "feasible")
        y = parameter_less(y, CV)
        yy = y.flatten()
        x = pop.get("X")

        x_mu, f_mu = self.mutate(x, yy, a)
        x_cr, p_cr = self.crossover(x_mu, x)
        x_cr = self.bound(x_cr, x)

        self.off = pop.new("X", x_cr)
        # evaluate the results
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        _y, _CV, _feasible = self.off.get("F", "CV", "feasible")
        _y = parameter_less(_y, _CV)
        # find the individuals which are indeed better
        is_better = np.where((_y <= y)[:, 0])[0]

        # replace the individuals in the population
        pop[is_better] = self.off[is_better]

        # update self.mu and self.median
        f = np.empty((0,))  # set of all successful mutation factors
        p = np.empty((0,))  # set of all successful crossover probabilities
        f = np.hstack((f, f_mu[is_better])) # archive of the successful mutation factor
        p = np.hstack((p, p_cr[is_better])) # archive of the successful crossover probability
        if len(p) != 0:  # for mean update of normal distribution
            self.mu = (1.0 - self.c) * self.mu + self.c * np.mean(p)
        if len(f) != 0:  # for location update of Cauchy distribution
            self.median = (1.0 - self.c) * self.median + self.c * np.sum(np.power(f, 2)) / np.sum(f)

        a = np.vstack((a, x[is_better]))  # archive of the inferior solution
        # randomly remove solutions to keep the archive size fixed
        if len(a) > self.pop_size:
            a = np.delete(a, np.random.choice(len(a), (len(a) - self.pop_size,), False), 0)

        # store the population in the algorithm object
        self.pop = pop
        self.a = a


    def mutate(self, x=None, y=None, a=None):
        pop_size = len(x)
        x_mu = np.empty((pop_size,  self.problem.n_var))  # mutated population
        f_mu = np.empty((pop_size,))  # mutated mutation factors
        order = np.argsort(y)[:int(np.ceil(self.p*pop_size))]  # index of the [100*p]% best individuals
        x_p = x[np.random.choice(order, (pop_size,))]
        x_un = np.vstack((np.copy(x), a))  # archive
        for k in range(pop_size):
            f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = np.random.choice([i for i in range(pop_size) if i != k])
            r2 = np.random.choice([i for i in range(len(x_un)) if i != k and i != r1])
            x_mu[k] = x[k] + f_mu[k]*(x_p[k] - x[k]) + f_mu[k]*(x[r1] - x_un[r2])
        return x_mu, f_mu

    def crossover(self, x_mu=None, x=None):
        pop_size = len(x)
        x_cr = np.copy(x)
        p_cr = np.random.normal(self.mu, 0.1, (pop_size,))  # crossover probabilities
        # truncate to [0, 1]
        p_cr = np.minimum(np.maximum(p_cr, 0.0), 1.0)
        for k in range(pop_size):
            i_rand = np.random.randint(self.problem.n_var)
            for i in range(self.problem.n_var):
                if (i == i_rand) or (np.random.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr
