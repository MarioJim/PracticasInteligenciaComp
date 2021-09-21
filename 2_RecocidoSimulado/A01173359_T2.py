import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class MultipleTravelingSalesmen:
    def __init__(self, coords: np.ndarray, n_salesmen: int = 2,
                 alpha: float = 0.8, beta: float = 1.5, n_batches: int = 200,
                 n_iterations: int = 500, min_accepted: float = 0.8,
                 max_batches_with_same_solution: int = 20):
        self.coords = coords
        self.distances = pairwise_distances(coords)
        self.n_salesmen = n_salesmen
        self.alpha = alpha
        self.beta = beta
        self.n_batches = n_batches
        self.n_iterations = n_iterations
        self.min_accepted = min_accepted
        self.max_batches_with_same_solution = max_batches_with_same_solution

        self.init_temperature()

    def init_temperature(self):
        print("**** Initialize temperature ****")
        self.temperature = 0.1
        solution = self.mutate_solution()
        n_accepted = 0
        iter = 1

        while n_accepted / self.n_iterations < self.min_accepted:
            solution, n_accepted = self.run_batch(solution)
            print("Batch {}:   T {:.2f}  \tacc {:.2f}".format(
                iter, self.temperature, n_accepted / self.n_iterations))
            iter += 1
            if iter == 200:
                raise RuntimeError(
                    "Temp took more than 200 iterations to initialize")
            self.temperature *= self.beta

    def evaluate_solution(self, solution: np.ndarray) -> float:
        rows = np.arange(solution.shape[0])
        return np.sum(self.distances[rows, solution])

    def generate_random_swap(self) -> Tuple[int, int]:
        sample_space = np.arange(self.distances.shape[0])
        samples = np.random.choice(sample_space, 2)
        return (samples[0], samples[1])

    def mutate_solution(self, original_solution: Optional[np.ndarray] = None) -> np.ndarray:
        if original_solution is None:
            solution = np.arange(self.distances.shape[0])
            np.random.shuffle(solution)
        else:
            solution = original_solution.copy()

        i, j = self.generate_random_swap()
        solution[i], solution[j] = solution[j], solution[i]

        while (original_solution == solution).all() or not self.is_solution_valid(solution):
            i, j = self.generate_random_swap()
            solution[i], solution[j] = solution[j], solution[i]

        return solution

    def calculate_cycle_number_map(self, solution: np.ndarray) -> np.ndarray:
        solution = solution.copy()
        for _ in range(solution.shape[0]):
            np.minimum(solution, solution[solution], solution)
        _, cycle_num_map = np.unique(solution, return_inverse=True)
        return cycle_num_map

    def is_solution_valid(self, solution: np.ndarray) -> bool:
        # Check number of cycles == number of salesmen
        n_cycles = self.calculate_cycle_number_map(solution).max() + 1
        return n_cycles == self.n_salesmen

    def run_batch(self, starting_solution: np.ndarray) -> Tuple[np.ndarray, int]:
        """ Cadena de Markov """
        best_solution = starting_solution
        best_evaluation = self.evaluate_solution(best_solution)
        n_accepted = 0

        for _ in range(self.n_iterations):
            new_solution = self.mutate_solution(best_solution)
            new_evaluation = self.evaluate_solution(new_solution)
            # Accepted because it's better
            if new_evaluation <= best_evaluation:
                best_solution = new_solution
                best_evaluation = new_evaluation
                n_accepted += 1
                continue

            random_probability = np.random.random()
            delta_eval = abs(new_evaluation - best_evaluation)
            delta_probability = math.exp(- delta_eval / self.temperature)
            # Accepted because of random probability
            if random_probability < delta_probability:
                best_solution = new_solution
                best_evaluation = new_evaluation
                n_accepted += 1

        return (best_solution, n_accepted)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        print("**** Run the algorithm {} times ****".format(self.n_batches))
        best_solution = self.mutate_solution()
        best_eval_history = np.empty(self.n_batches)
        best_eval_history[0] = self.evaluate_solution(best_solution)
        batches_with_same_solution = 0

        for i in range(1, self.n_batches):
            new_solution, n_accepted = self.run_batch(best_solution)
            best_eval_history[i] = self.evaluate_solution(new_solution)
            print("Batch {:4}: T e^{:.3f}\tacc {:.3f} eval {:.7f} ".format(
                i + 1, math.log(self.temperature),
                n_accepted / self.n_iterations, best_eval_history[i]))
            if (best_solution == new_solution).all():
                batches_with_same_solution += 1
            else:
                best_solution = new_solution
                batches_with_same_solution = 0
            if batches_with_same_solution == self.max_batches_with_same_solution:
                print("Early termination because {} consecutive batches returned the same solution"
                      .format(batches_with_same_solution))
                best_eval_history = best_eval_history[:i+1]
                break
            self.temperature *= self.alpha

        return (best_solution, best_eval_history)

    def graph_solution(self, solution: np.ndarray):
        solution_idxs = np.arange(len(solution))
        x1s = self.coords[solution_idxs, 0]
        y1s = self.coords[solution_idxs, 1]
        x2s = self.coords[solution, 0]
        y2s = self.coords[solution, 1]

        cycle_num_map = self.calculate_cycle_number_map(solution)
        n_cycles = cycle_num_map.max() + 1
        colormap = plt.get_cmap("hsv")
        colors = [colormap(i / n_cycles) for i in cycle_num_map]

        plt.title('Traveling Salesmen Map')
        for x1, y1, x2, y2, color in zip(x1s, y1s, x2s, y2s, colors):
            plt.plot([x1, x2], [y1, y2], 'o-', c=color, mfc=color)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    def graph_evaluation_history(self, evaluation_history: np.ndarray):
        history_size = evaluation_history.shape[0]
        plt.title('Best evaluation curve')
        plt.plot(range(history_size), evaluation_history,
                 color='red', marker='*')
        plt.ylim(bottom=0)
        plt.xlabel('Markov Chains')
        plt.ylabel('Sum of distances from best found')


if __name__ == "__main__":
    n_cities = 30
    coords = np.random.rand(n_cities, 2)
    n_salesmen = 6

    tsp = MultipleTravelingSalesmen(coords, n_salesmen)
    best_solution, best_evaluation_history = tsp.run()
    tsp.graph_solution(best_solution)
    plt.savefig("map.png")
    plt.clf()
    tsp.graph_evaluation_history(best_evaluation_history)
    plt.savefig("curve.png")
