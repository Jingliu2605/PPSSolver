# Jing Liu
import numpy as np

from pymoo.model.sampling import Sampling
from pymoo.util.misc import cdist

class FeedSampling(Sampling):
    """
    Sampling given
    """
    def _do(self, problem, n_samples, **kwargs):
