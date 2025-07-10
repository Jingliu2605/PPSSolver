# cython: boundscheck=False, wraparound=False, profile=False
import numpy as np
cimport numpy as np

from problem.portfolio import Portfolio
from pymoo.model.crossover import Crossover

cimport numpy as np
import numpy as np

from problem.portfolio import Portfolio
from pymoo.model.crossover import Crossover


class PortfolioShuffledCrossover(Crossover):

    def __init__(self, greedy=True, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.greedy = greedy

    def _do(self, problem, X, **kwargs):
        cdef int n_matings, n_var, j, i, index

        n_matings = X.shape[1]
        n_var = X.shape[2]

        # start point of crossover
        _X = np.copy(X)

        for j in range(n_matings):
            p1 = _X[0][j]
            p2 = _X[1][j]

            # preallocate as maximum possible size
            pairs = [(None, None)] * (n_var * 2)
            index = 0

            # build list of (unique) tuples containing (project, start_time) from both parents
            for i in range(n_var):
                if p1[i] > 0:
                    pairs[index] = (i, p1[i])
                    index += 1
                if p2[i] > 0 and p2[i] != p1[i]:
                    pairs[index] = (i, p2[i])
                    index += 1

            # convert to numpy array, using only the indices that were populated
            np_pairs = np.array(pairs[:index])

            _X[0][j] = self._build_child(np_pairs, problem.instance)
            _X[1][j] = self._build_child(np_pairs, problem.instance)

        return _X

    def _build_child(self, np.int_t[:, :] pairs, instance):
        cdef int i, index, time
        cdef np.int_t[:] unscheduled
        portfolio = Portfolio(instance.num_projects, instance.budget_window,
                              instance.planning_window, instance.discount_rate,
                              instance.capability_stream_budgets.shape[0])

        # TODO: consider creating an index permutation, rather than shuffling the pairs
        # TODO: order prerequisites before any projects that come after them?
        #np.random.permutation(pairs.shape[0])
        np.random.shuffle(pairs)

        for i in range(pairs.shape[0]):
            index = pairs[i][0]
            time = pairs[i][1]
            # skip project if it has been scheduled (by way of other parent)
            if portfolio.scheduled(index):
                continue

            portfolio.add_if_feasible(index, time, instance)

        if not self.greedy:
            return portfolio.result

        result = portfolio.result
        # find indices that are zero and shuffle
        unscheduled = np.nonzero(result==0)[0]
        np.random.shuffle(unscheduled)
        # loop through unscheduled and randomly assign to earliest or latest feasible time, or ignore if infeasible
        for i in range(unscheduled.shape[0]):
            index = unscheduled[i]
            if np.random.random() < 0.5:
                portfolio.add_earliest_feasible(index, instance.projects[index], instance.budget, instance.capability_stream_budgets,
                                                     instance.planning_window)
            else:
                portfolio.add_latest_feasible(index, instance.projects[index], instance.budget, instance.capability_stream_budgets,
                                                   instance.planning_window)

        return portfolio.result

    def __str__(self):
        return f"PSX (greedy={self.greedy})"

    def __repr__(self):
        return self.__str__()
