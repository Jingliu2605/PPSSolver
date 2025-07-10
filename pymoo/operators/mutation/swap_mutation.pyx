# cython: profile=False
import numpy as np

from pymoo.model.mutation import Mutation


class SwapMutation(Mutation):
    """
    For permutation representation
    """
    def __init__(self, prob=0.1, swaps=1):
        super().__init__()
        self.prob = prob
        self.swaps = swaps

    def _do(self, problem, X, **kwargs):
        _X = np.copy(X)
        # TODO: vectorize as much as possible
        # TODO: consider the probability of swap as a per gene probability
        # do_mutation = np.random.random(X.shape[0]) < self.prob
        if isinstance(self.prob, float):
            print("float")
            self.prob = np.tile(self.prob, X.shape[0])
        for i in range(X.shape[0]):
            if np.random.random() < self.prob[i]:  # do_mutation[i]: #
                mutant = _X[i]
                length = mutant.shape[0]
                for j in range(self.swaps):
                    ind1, ind2 = np.random.choice(range(length), 2, replace=False)
                    mutant[ind1], mutant[ind2] = mutant[ind2], mutant[ind1]
        return _X

    def __str__(self):
        return f"Swap Mutation (p={self.prob}, swaps={self.swaps})"

    def __repr__(self):
        return self.__str__()
