from problem.portfolio import build_from_array
from pymoo.model.repair import Repair


class PrerequisiteRepair(Repair):

    def _do(self, problem, pop, **kwargs):

        X = pop.get("X")  # get solution vectors

        for i in range(len(X)):
            x = X[i]
            portfolio, violations = build_from_array(x, problem.instance)
            if violations['prereq_viols'] == 0:
                return pop  # no mutual exclusion violations, return the population without modification

            for p in range(len(x)):
                if x[p] > 0:  # if p is selected
                    # loop through exclusions
                    for index in range(problem.projects[p].prerequisite_list.shape[0]):
                        prerequisite = problem.projects[p].prerequisite_list[index]
                        # if prerequisite is selected, but uncompleted
                        if x[prerequisite] > 0 and x[prerequisite] + problem.projects[prerequisite].duration < x[p]:
                            x[p] = 0  # deselect the specified project

        return pop.new("X", X)
