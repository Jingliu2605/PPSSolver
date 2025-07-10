import numpy as np

from pymoo.model.mutation import Mutation


class ScrambleMutation(Mutation):

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def _do(self, problem, x, **kwargs):
        cdef int i, ind1, ind2
        _X = np.copy(x)
        # TODO: vectorize?
        for i in range(x.shape[0]):
            if np.random.random() < self.prob:
                mutant = _X[i]
                #TODO: reduce complexity at expense of possible overlap
                ind1, ind2 = np.random.choice(mutant.shape[0], 2, replace=False)
                #ind1 = np.random.randint(0, mutant.shape[0])
                #ind2 = np.random.randint(0, mutant.shape[0])
                if ind1 > ind2:
                    ind1, ind2 = ind2, ind1
                #use shuffle to randomly shuffle the slice in place
                np.random.shuffle(mutant[ind1:ind2])
                #mutant[ind1:ind2] = np.random.permutation(mutant[ind1:ind2])
        return _X

    def __str__(self):
        return f"Scramble Mutation (p={self.prob})"

    def __repr__(self):
        return self.__str__()
