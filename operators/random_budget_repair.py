import numpy as np

from problem.portfolio import build_from_array
from pymoo.model.repair import Repair


class RandomBudgetRepair(Repair):
    def __init__(self, prob_repair=1, real_flag=0, **kwargs):
        self.prob_repair = prob_repair
        self.real_flag = real_flag

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")  # get solution vectors
        # iterate through all solution vectors
        for i in range(len(X)):
            # old = X[i]
            if self.real_flag:
                x = np.round(X[i]).astype(int)
            else:
                x = X[i]
            portfolio, violations = build_from_array(x, problem.instance)
            budget_viols = violations["budget_viols"]
            viol_sum = np.sum(budget_viols)
            # exit if no violations or probability check fails
            if viol_sum <= 0 or np.random.random() > self.prob_repair:
                continue

            viol_times = np.nonzero(violations["budget_viols"])[0]

            # find and remove projects that run during these times to remove them
            for t in viol_times:
                amount = violations["budget_viols"][t]
                # find indices of projects that are running in time t
                indices = [i for i in range(problem.instance.num_projects) if
                           0 < portfolio.result[i] < t <= portfolio.result[i]
                           + problem.instance.projects[i].duration]
                shuffled = np.random.permutation(indices)
                for p in shuffled:
                    portfolio.remove_from_portfolio(p, problem.instance.projects[p])
                    x[p] = 0
                    violations = portfolio.constraint_violations(problem.instance)
                    if violations["budget_viols"][t] <= 0:
                        break
            X[i] = x

            # viol_times = np.nonzero(violations["budget_viols"])

        pop.set("X", X)
        return pop
