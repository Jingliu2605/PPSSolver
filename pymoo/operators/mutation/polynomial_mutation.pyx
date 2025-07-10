import numpy as np

from pymoo.model.mutation import Mutation
from pymoo.operators.repair.out_of_bounds_repair import set_to_bounds_if_outside_by_problem


class PolynomialMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, x, **kwargs):

        x = x.astype(np.float)
        y = np.full(x.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = np.random.random(x.shape) < self.prob

        y[:, :] = x

        xl = np.repeat(problem.xl[None, :], x.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], x.shape[0], axis=0)[do_mutation]

        x = x[do_mutation]

        delta1 = (x - xl) / (xu - xl)
        delta2 = (xu - x) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(x.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(x.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = x + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        y = set_to_bounds_if_outside_by_problem(problem, y)

        return y

    def __str__(self):
        return f"Polynomial Mutation (p={self.prob}, eta={self.eta})"

    def __repr__(self):
        return self.__str__()
