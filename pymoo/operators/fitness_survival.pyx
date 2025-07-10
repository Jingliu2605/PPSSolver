import numpy as np

from pymoo.model.survival import Survival


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, problem, pop, n_survive, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective fitness!")

        return pop[np.argsort(F[:, 0])[:n_survive]]
