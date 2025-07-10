import numpy as np

from problem.portfolio import build_from_array
from solvers.simulated_annealing import SimulatedAnnealing


class StartTimeAnnealer(SimulatedAnnealing):
    """
    Simulated annealing using start time representation
    """

    def __init__(self, initial_state, instance, random_seed):
        self.instance = instance
        super(StartTimeAnnealer, self).__init__(initial_state, random_seed=random_seed)
        self.updates = self.steps / 100

    def move(self):
        """
        Swaps the start times of two projects or randomly resets one projects start time.
        """

        if np.random.random() < 0.5:
            a, b = np.random.randint(0, self.state.shape[0], 2)
            self.state[a], self.state[b] = self.state[b], self.state[a]
        else:
            a = np.random.randint(0, self.state.shape[0])
            self.state[a] = np.random.randint(0, self.instance.planning_window + 1)
        #return self.energy() - initial_energy

    def energy(self):
        """Calculates the total value of the portfolio, returning a positive value if there are violations."""
        portfolio, violations = build_from_array(self.state, self.instance)

        #if there are violations, return them as a positive values
        viol_sum = violations['all_viols'].sum()
        #violations["budget_viols"].sum() + violations["stream_viols"].sum() + \
                   #violations["initiation_viols"].sum() + violations["prereq_viols"] + violations["exclusion_viols"]
        if viol_sum > 0:
            return viol_sum

        return -portfolio.value
