import numpy as np

from executor import generate_seeds
from operators.my_ga import MyGA
from operators.seeded_sampling import SeededSampling
from problem.hierarchical_selection_problem import HierarchicalSelectionProblem
from problem.portfolio import Portfolio, build_from_array
from problem.portfolio_selection_instance import PortfolioSelectionInstance
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.operators.mutation.bit_flip_mutation import BitFlipMutation
from pymoo.optimize import minimize


class HierarchicalPlanning:
    """
    Use Hierarchical Planning to generate a feasible solution
    """

    def __init__(self, instance, pop_size, evals_per_year, seed=1):
        self.name = "Hierarchical Planning"
        self.instance = instance
        self.pop_size = pop_size
        self.evals_per_year = evals_per_year
        self.seed = seed
        np.random.seed(seed)

    def solve(self):
        portfolio = Portfolio(self.instance.num_projects, self.instance.budget_window, self.instance.planning_window)

        instance = self.instance
        available = np.nonzero(portfolio.result == 0)[0]  # self.instance.projects
        for t in range(1, self.instance.planning_window + 1):
            # TODO: this needs to more adequately address constraints
            seeds = generate_seeds(instance, self.pop_size, self.seed)

            # extract only the projects for the first year as the initial seeds
            ga_seeds = np.ndarray((seeds.shape[0], instance.num_projects))
            for i in range(seeds.shape[0]):
                ga_seeds[i] = seeds[i] == 1

            # generate a GA with the given problem formulation
            result = self.run_ga(instance, ga_seeds)
            selected = result.X.astype(int)

            # loop through result, find original project, add to portfolio
            for i in range(selected.shape[0]):
                if selected[i]:
                    project_index = available[i]
                    portfolio.add_to_portfolio(project_index, t, self.instance.projects[project_index])

            # get list of unselected projects
            available = np.nonzero(portfolio.result == 0)[0]
            # calculate the remaining budget after this portfolio is selected
            available_budget = self.instance.budget - portfolio.cost
            # define a new planning window
            planning_window = instance.planning_window - 1
            # define the simplified selection instance
            instance = PortfolioSelectionInstance(self.instance.projects[available], available_budget, planning_window, instance.discount_rate)

        return portfolio

    def run_ga(self, instance, seeds):
        method = MyGA(pop_size=self.pop_size, sampling=SeededSampling(seeds), crossover=HalfUniformCrossover(),
                      mutation=BitFlipMutation(), eliminate_duplicates=True, return_least_infeasible=True,
                      verbose=True)

        problem = HierarchicalSelectionProblem(instance)

        result = minimize(problem, method, termination=('n_eval', self.evals_per_year), seed=None, save_history=True,
                          verbose=True)

        return result
