from pymoo.model.duplicate import DuplicateElimination


class NoDuplicateElimination(DuplicateElimination):

    def do(self, pop, *args, **kwargs):
        return pop
