import numpy as np

from problem.portfolio import build_from_array
from pymoo.model.repair import Repair


class MutualExclusionRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        kwargs.setdefault('selection', 'random')
        selection = kwargs['selection']

        X = pop.get("X")  # get solution vectors

        for i in range(len(X)):
            x = X[i]
            portfolio, violations = build_from_array(x, problem.instance)
            if violations['exclusion_viols'] == 0:
                return pop  # no mutual exclusion violations, return the population without modification

            for p in range(len(x)):
                if x[p] > 0:  # if p is selected
                    # loop through exclusions
                    for index in range(problem.projects[p].exclusion_list.shape[0]):
                        exclusion = problem.projects[p].exclusion_list[index]
                        if x[exclusion] > 0:
                            if selection == 'random':  # select a random one
                                remove = p if np.random.random() < 0.5 else exclusion
                            elif selection == 'value':  # select the project with the lesser value
                                remove = p if problem.projects[p].value < problem.projects[
                                    exclusion].value else exclusion
                            elif selection == 'latest':  # select the project with the later starting time
                                remove = p if x[p] > x[exclusion] else exclusion

                            x[remove] = 0  # deselect the specified project

        return pop.new("X", X)
