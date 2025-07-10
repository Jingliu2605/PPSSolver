import numpy as np

from pymoo.model.crossover import Crossover


cpdef remove_from_adj_list(H, val):
    for e in list(H[val]):
        H[e].remove(val)
    del H[val]


cpdef has_duplicates(x):
    H = set()
    for v in x:
        if v in H:
            return True
        H.add(v)
    return False


cpdef erx(a, b):
    """
    http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/EdgeRecombinationCrossoverOperator.aspx
    Algorithm Pseudo Code:
    1. X = the first node from a random parent.
    2. While the CHILD chromo isn't full, Loop:
        - Append X to CHILD
        - Remove X from Neighbor Lists
        if X's neighbor list is empty:
           - Z = random node not already in CHILD
        else
           - Determine neighbor of X that has fewest neighbors
           - If there is a tie, randomly choose 1
           - Z = chosen node
        X = Z
     """

    assert len(a) == len(b)

    # calculate the edge matrix considering both permutation
    H = calc_adjency_matrix(a)
    H = calc_adjency_matrix(b, H=H)

    # randomly select the first node
    _next = np.random.choice(list(H.keys()))

    y = []
    while True:

        # append to the child
        y.append(_next)

        # break if the child was successfully created.
        if len(y) == len(a):
            break

        # get the neighbors to consider and remove them from the lists
        neighbors = list(H[_next])
        remove_from_adj_list(H, _next)

        # if the current node does not have any neighbors
        if len(neighbors) == 0:
            _next = np.random.choice(list(H.keys()))

        # otherwise search in the neighbors for a node with the fewest neighbors
        else:
            # search for the one with minimum neighbors
            n_neighbors = [len(H[e]) for e in neighbors]
            min_n_neighbors = min(n_neighbors)
            _next = [neighbors[k] for k in range(len(neighbors)) if n_neighbors[k] == min_n_neighbors]

            # break the tie if they might have the same number of neighbors
            _next = np.random.choice(_next)

    return y


class EdgeRecombinationCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 1, **kwargs)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            a, b = X[:, i, :]
            Y[0, i, :] = erx(a, b)

        return Y


cpdef number_to_letter(n):
    return chr(ord('@') + n)


cpdef numbers_to_letters(numbers):
    return [number_to_letter(n) for n in numbers]


cpdef letter_to_number(char):
    return ord(char.lower()) - 96


cpdef letters_to_numbers(letters):
    return np.array([letter_to_number(char) for char in letters])


cpdef calc_adjency_matrix(x, H=None):
    H = {} if H is None else H

    for k in range(len(x)):
        prev = (k - 1) % len(x)
        succ = (k + 1) % len(x)

        if x[k] not in H:
            H[x[k]] = set()
        H[x[k]].update([x[prev], x[succ]])

    return H
