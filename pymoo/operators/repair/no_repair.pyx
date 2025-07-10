from pymoo.model.repair import Repair


class NoRepair(Repair):
    """
    A dummy class which can be used to simply do no repair.
    """

    def _do(self, problem, pop, **kwargs):
        return pop
