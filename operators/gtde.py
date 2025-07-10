# Jing Liu
import numpy as np
import time
import random
import math

from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.model.algorithm import filter_optimum
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.util.misc import parameter_less
from operators.my_ga import basic_stats
from operators.iteration_data import IterationData
from pymoo.operators.fitness_survival import FitnessSurvival
from pymoo.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from pymoo.util.display import SingleObjectiveDisplay

class MyGTDE(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=LatinHypercubeSampling(iterations=100, criterion="maxmin"),
                 exnum=400,
                 f=0.7,
                 sig_f=0.5,
                 CR=0.5,
                 sig_CR=0.5,
                 exF=0.5,
                 sig_exF=0.1,
                 meanpj=0.01,
                 stdpj=0.01,
                 eliminate_duplicates=True,
                 verbose=False,
                 display=SingleObjectiveDisplay(resolution=100),
                 **kwargs):
        self.pop_size = pop_size
        self.exnum = exnum
        self.f = f
        self.sig_f = sig_f
        self.CR = CR
        self.sig_CR = sig_CR
        self.exF = exF
        self.sig_exF = sig_exF
        self.meanpj = meanpj
        self.stdpj = stdpj
        self.bestnum = 0
        self.besty = float('inf')
        self.used_FES = 0
        self.copy_FES = 0
        # self.ubound, self.lbound = 0.0, 0.0
        # self.obj_function = None
        self.info = None
        self.Pm = 0.01
        self.phase = 0
        self.v1 = None
        self.v2 = None
        self.s = None

        self.default_termination = MaximumGenerationTermination(500)
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         survival=FitnessSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         display=display,
                         verbose=verbose,
                         **kwargs)

    def _next(self):
        # retrieve the current population
        pop = self.pop

        # get the vectors from the population
        F, CV, feasible = pop.get("F", "CV", "feasible")
        F = parameter_less(F, CV)
        self.best_num = np.argmin(F[:, 0])
        self.besty = F[self.best_num]
        x = pop.get("X")
        NP, Dim = x.shape
        v = np.zeros((NP, Dim))
        u = np.zeros((NP, Dim))
        exv = np.zeros((self.exnum, Dim))
        exu = np.zeros((self.exnum, Dim))
        # mut_cro_sel
        for t in range(NP):
            if t != self.bestnum:
                while True:
                    r1, r2, r3 = random.randint(0, NP - 1), random.randint(0, NP - 1), random.randint(0, NP - 1)
                    if r1 != r2 and r1 != r3 and r1 != t and r2 != r3 and r2 != t and r3 != t:
                        break

                e_f = 0.0
                while e_f <= 0:
                    e_f = self.norm_dist(self.f, self.sig_f)
                if e_f > 1:
                    e_f = 1

                e_cr = 0.0
                while e_cr <= 0:
                    e_cr = self.norm_dist(self.CR, self.sig_CR)
                if e_cr > 1:
                    e_cr = 1

                r = random.randint(0, Dim - 1)
                for j in range(Dim):
                    if j == r or random.random() < e_cr:
                        v[t][j] = x[t][j] + e_f * (x[self.bestnum][j] - x[t][j]) + e_f * (x[r1][j] - x[r2][j])
                        u[t][j] = v[t][j]
                    else:
                        u[t][j] = x[t][j]

                self.off = pop.new("X", [u[t]])
                self.off = self.repair.do(self.problem, self.off)
                self.evaluator.eval(self.problem, self.off, algorithm=self)
                _F, _CV, _feasible = self.off.get("F", "CV", "feasible")
                _F = parameter_less(_F, _CV)
                if _F < F[t]:
                    x[t] = u[t]
                    pop[t] = self.off[0]
                    F[t] = _F
                    if _F < self.besty:
                        self.besty = _F
                        self.bestnum = t

                self.used_FES += 1
                self.copy_FES += 1

            else:  # best individual
                for i in range(self.exnum):
                    bottleneck = [False] * Dim

                    pj = 0.0
                    while pj <= 0:
                        pj = self.norm_dist(self.meanpj, self.stdpj)
                    if pj > 1:
                        pj = 1

                    for j in range(Dim):
                        if random.random() < pj:
                            bottleneck[j] = True

                    while True:
                        r1, r2 = random.randint(0, NP - 1), random.randint(0, NP - 1)
                        if r1 != r2 and r1 != t and r2 != t:
                            break

                    e_f = 0.0
                    while e_f <= 0:
                        e_f = self.norm_dist(self.exF, self.sig_exF)
                    if e_f > 1:
                        e_f = 1

                    for j in range(Dim):
                        if bottleneck[j]:  # bottleneck
                            if random.random() < self.Pm:  # DE/r-best/1
                                ran = random.uniform(self.problem.xl[j], self.problem.xu[j])
                                exv[i][j] = x[t][j] + e_f * (x[r1][j] - ran)
                            else:
                                exv[i][j] = x[t][j] + e_f * (x[r1][j] - x[r2][j])
                            exu[i][j] = exv[i][j]
                        else:
                            exu[i][j] = x[t][j]

                    self.off = pop.new("X", [exu[i]])
                    self.off = self.repair.do(self.problem, self.off)
                    self.evaluator.eval(self.problem, self.off, algorithm=self)
                    _F, _CV, _feasible = self.off.get("F", "CV", "feasible")
                    _F = parameter_less(_F, _CV)
                    if _F < F[t]:
                        x[t] = exu[i]
                        pop[t] = self.off[0]
                        F[t] = _F
                        if _F < self.besty:
                            self.besty = _F
                            self.bestnum = t

                    self.used_FES += 1
                    self.copy_FES += 1
        self.pop = pop

    def norm_dist(self, mean, sd):
        if self.phase == 0:
            while True:
                u1 = random.random()
                u2 = random.random()
                self.v1 = 2 * u1 - 1
                self.v2 = 2 * u2 - 1
                self.s = self.v1 * self.v1 + self.v2 * self.v2
                if self.s < 1 and self.s != 0:
                    break
            xx = self.v1 * math.sqrt(-2 * math.log(self.s) / self.s)
        else:
            xx = self.v2 * math.sqrt(-2 * math.log(self.s) / self.s)

        self.phase = 1 - self.phase

        return xx * sd + mean

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
            # print(exec_time)
            # if round(exec_time % 60) <= 1:
            self.time_history.append([self.n_gen, self.evaluator.n_eval, self.opt.get("F"), exec_time])
