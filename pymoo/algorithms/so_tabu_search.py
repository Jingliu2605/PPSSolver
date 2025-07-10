from pymoo.algorithms.so_local_search import LocalSearch
from pymoo.util.display import SingleObjectiveDisplay


class TabuSearch(LocalSearch):

    def __init__(self, num_candidates, display=SingleObjectiveDisplay(), **kwargs):
        super.__init__(n_max_candidates=num_candidates, display=display, **kwargs)

    def _next(self):
        pass
