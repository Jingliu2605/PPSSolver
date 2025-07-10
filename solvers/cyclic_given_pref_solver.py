from solvers.cyclic_base import CyclicBaseSolver


class CyclicGivenPrefSolver(CyclicBaseSolver):
    """
    Generate a feasible solution using cyclic project selection.

    While not completed:
        For each time step:
            Select feasible project by cost (if available)
            Continue to next time step
        If no projects selected, mark completed
    """

    def __init__(self, instance, preferences, seed=1):
        super().__init__(instance, seed)
        self.name = "Cyclic (Given Preferences)"
        self.preferences = preferences

    def _preferences(self):
        return self.preferences
