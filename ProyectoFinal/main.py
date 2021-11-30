import random
import sys
from typing import Tuple

from deap.tools import mutShuffleIndexes, cxPartialyMatched
import matplotlib.pyplot as plt
import numpy as np

from tsp import TravelingSalesmanProblem


def mutDisplacement(indiv: list) -> Tuple[list]:
    p, sz = sorted(random.sample(range(len(indiv) // 3), 2))
    indiv[p:p+sz], indiv[p+sz:p+sz+sz] = indiv[p+sz:p+sz+sz], indiv[p:p+sz]
    return indiv,


def cxVotingRecombination(ind1: list, ind2: list) -> Tuple[list, list]:
    not_same = set()
    for v1, v2 in zip(ind1, ind2):
        if v1 != v2:
            not_same.add(v1)
            not_same.add(v2)

    not_same1 = list(not_same)
    random.shuffle(not_same1)
    not_same2 = list(not_same)
    random.shuffle(not_same2)

    for i, (v1, v2) in enumerate(zip(ind1, ind2)):
        if v1 in not_same:
            ind1[i] = not_same1.pop()
            ind2[i] = not_same2.pop()

    return ind1, ind2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter the number of cities to generate")
        exit(1)

    random.seed(0)

    # num_cities = int(sys.argv[1])
    # coords = np.random.uniform(-100, 100, (num_cities, 2))
    # np.savetxt("coords.txt", coords)
    coords = np.loadtxt("coords.txt")

    algorithm = "simple"
    # algorithm = "plus"
    # algorithm = "comma"
    def mutation_f(ind): return mutShuffleIndexes(ind, 0.05)
    # def mutation_f(ind): return mutDisplacement(ind)
    def crossover_f(ind1, ind2): return cxPartialyMatched(ind1, ind2)
    # def crossover_f(ind1, ind2): return cxVotingRecombination(ind1, ind2)

    tsp_instance = TravelingSalesmanProblem(
        coords, algorithm, mutation_f, crossover_f)

    means, stds, mins, hof = tsp_instance.execute()
    tsp_instance.print_hall_of_fame(hof)
    tsp_instance.plot_generations(means, stds, mins)
    plt.savefig("generations.png")
    tsp_instance.plot_solution(hof.items[0])
    plt.savefig("solution.png")
