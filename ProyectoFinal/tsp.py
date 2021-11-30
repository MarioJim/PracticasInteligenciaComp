from typing import Callable, Tuple
import random

from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class TravelingSalesmanProblem:
    def __init__(self, coords: np.ndarray, algorithm: str,
                 mutation_f: Callable, crossover_f: Callable,
                 mutation_p: float = 0.3, crossover_p: float = 0.7,
                 generations: int = 1000,
                 population_size: int = 300, mu: int = 100, lambda_: int = 200):
        self.coords = coords
        self.num_cities = coords.shape[0]
        self.distances = pairwise_distances(coords)

        possible_algs = ["simple", "plus", "comma"]
        if algorithm not in possible_algs:
            print(f"Parameter 'algorithm' must be one of {possible_algs}")
            exit(1)
        self.algorithm = algorithm

        self.mutate = mutation_f
        self.mate = crossover_f
        self.mutate_pb = mutation_p
        self.mate_pb = crossover_p
        self.generations = generations

        self.population_size = population_size
        self.mu = mu
        self.lambda_ = lambda_

    def eval_solution(self, solution: np.ndarray) -> float:
        """Calculates the total distance of a given path"""
        path_dist = np.sum(self.distances[solution[:-1], solution[1:]])
        back_dist = self.distances[solution[-1], solution[0]]
        return path_dist + back_dist

    def execute(self, num_items_hof: int = 5) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, tools.HallOfFame]:
        """Executes the chosen algorithm"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # Evolution parameters and functions
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mutate", self.mutate)
        toolbox.register("mate", self.mate)
        # Evaluation function
        toolbox.register("evaluate", lambda ind: (self.eval_solution(ind),))
        # Population and individual generation
        toolbox.register("indices", random.sample,
                         range(self.num_cities), self.num_cities)
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.indices)
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)

        population = toolbox.population(n=self.population_size)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)

        hof = tools.HallOfFame(num_items_hof)

        if self.algorithm == "simple":
            _, logbook = algorithms.eaSimple(
                population, toolbox, self.mate_pb, self.mutate_pb,
                self.generations, stats, hof)
        elif self.algorithm == "plus":
            _, logbook = algorithms.eaMuPlusLambda(
                population, toolbox, self.mu, self.lambda_, self.mate_pb,
                self.mutate_pb, self.generations, stats, hof)
        else:
            _, logbook = algorithms.eaMuCommaLambda(
                population, toolbox, self.mu, self.lambda_, self.mate_pb,
                self.mutate_pb, self.generations, stats, hof)

        return (logbook.select("avg"), logbook.select("std"),
                logbook.select("min"), hof)

    def print_hall_of_fame(self, hof: tools.HallOfFame):
        print("*** Hall of fame *** ")
        for idx, indiv in enumerate(hof.items):
            print(f"{idx + 1}. {indiv} = {self.eval_solution(indiv)}")

    def plot_generations(self, means: list, stds: list, mins: list):
        """
        Generates a graph with the average evolution
        of the generations
        """
        x = range(len(means))
        means = np.array(means)
        stds = np.array(stds)
        mins = np.array(mins)
        plt.figure()
        plt.title('Best evaluation curve')
        plt.plot(x, means, color='r', marker='*')
        plt.plot(x, means + stds, color='b', linestyle='-.')
        plt.plot(x, means - stds, color='b', linestyle='-.')
        plt.plot(x, mins, color='g', marker='*')
        plt.plot(x, np.minimum.accumulate(mins), color='c', marker=None)
        plt.xlabel('Generations')
        plt.legend(['average', 'average + std', 'average - std',
                    'best in generation', 'best found'])

    def plot_solution(self, solution: np.ndarray):
        """Generates a graph of a solution"""
        x1s = self.coords[solution[:-1], 0]
        y1s = self.coords[solution[:-1], 1]
        x2s = self.coords[solution[1:], 0]
        y2s = self.coords[solution[1:], 1]

        plt.figure()
        plt.title('Map - Traveling Salesman Problem')
        for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
            plt.plot([x1, x2], [y1, y2], 'bo-')
        plt.plot([self.coords[solution[-1], 0], self.coords[solution[0], 0]],
                 [self.coords[solution[-1], 1], self.coords[solution[0], 1]],
                 'bo-')
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
