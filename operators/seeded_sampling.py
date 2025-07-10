from pymoo.model.sampling import Sampling


class SeededSampling(Sampling):
    def __init__(self, seeds):
        self.seeds = seeds

    def _do(self, problem, n_samples, **kwargs):
        return self.seeds
