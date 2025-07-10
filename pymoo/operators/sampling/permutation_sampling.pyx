import numpy as np

from pymoo.model.sampling import Sampling


class PermutationSampling(Sampling):
    def __init__(self):
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        samples = np.ndarray((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            samples[i] = np.random.permutation(problem.n_var)

        return samples
