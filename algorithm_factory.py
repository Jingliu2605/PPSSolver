from solvers.cyclic_solver import CyclicSolver
from solvers.cyclic_ordering_solver import CyclicOrderingSolver
from solvers.pref_earliest_solver import PrefEarliestSolver
from solvers.pref_latest_solver import PrefLatestSolver
from solvers.random_cyclic_solver import RandomCyclicSolver
from solvers.random_earliest_solver import RandomEarliestSolver
from solvers.random_latest_solver import RandomLatestSolver
from solvers.random_solver import RandomSolver
from solvers.roulette_solver import RouletteSolver

from problem.enums import PortfolioOrdering


def get_solver_by_name(name, instance, seed):
    algorithms = {
        "randomsolver": (RandomSolver, {}),
        "cyclicsolver": (CyclicSolver, {}),
        "randomcyclicsolver": (RandomCyclicSolver, {}),
        "cyclicprefsolver": (CyclicOrderingSolver, dict(ordering=PortfolioOrdering.CV_RATIO)),
        "cyclichighvaluesolver": (CyclicOrderingSolver, dict(ordering=PortfolioOrdering.VALUE_DESC)),
        "cycliclowvaluesolver": (CyclicOrderingSolver, dict(ordering=PortfolioOrdering.VALUE_ASC)),
        "cyclichighcostsolver": (CyclicOrderingSolver, dict(ordering=PortfolioOrdering.COST_DESC)),
        "cycliclowcostsolver": (CyclicOrderingSolver, dict(ordering=PortfolioOrdering.COST_ASC)),
        "randomearliestsolver": (RandomEarliestSolver, {}),
        "randomlatestsolver": (RandomLatestSolver, {}),
        "prefearliestsolver": (PrefEarliestSolver, {}),
        "preflatestsolver": (PrefLatestSolver, {}),
        "roulettesolver": (RouletteSolver, {}),
        # "cyclicavgprefsolver": CyclicAvgPrefSolver
    }

    # TODO: add error checking for the name
    alg, kwargs = algorithms[name.lower()]
    return alg(instance, seed=seed, **kwargs)
