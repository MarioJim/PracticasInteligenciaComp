import sys

from deap import algorithms, base, creator, tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Ensuring an algorithm was selected
possible_algs = ["simple", "plus", "comma"]
if len(sys.argv) < 2 or sys.argv[1] not in possible_algs:
    print("Add an argument from the list:", possible_algs)
    exit(1)

# Load inputs
p = pd.read_csv('./p08_p.txt', header=None, names=["p"])["p"]
w = pd.read_csv('./p08_w.txt', header=None, names=["w"])["w"]
wmax = 6404180

# Params for the genetic algorithm
cxpb = 0.7
mutpb = 0.3
ngen = 1000
mu = 10
lambda_ = 20

# Setup the objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Evolution params
toolbox.register("select", tools.selRoulette)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("mate", tools.cxOnePoint)
# Evaluation params
toolbox.register("evaluate", lambda i: (np.sum(i * p),))
toolbox.decorate("evaluate",
                 tools.DeltaPenalty(lambda i: np.sum(i * w) <= wmax, 0))
# Population and individual params
toolbox.register("attribute", np.random.randint, low=0, high=2)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=24)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Stats for every execution
avgs_list = []
stds_list = []
root_hof = tools.HallOfFame(5)

# Execute the algorithm 10 times
for i in range(10):
    print(f"Execution {i + 1}")
    popul = toolbox.population(n=10)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    hof = tools.HallOfFame(5)

    logbook = None
    if sys.argv[1] == possible_algs[0]:
        _, logbook = algorithms.eaSimple(
            popul, toolbox, cxpb, mutpb, ngen, stats, hof, False)
    elif sys.argv[1] == possible_algs[1]:
        _, logbook = algorithms.eaMuPlusLambda(
            popul, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats, hof, False)
    else:
        _, logbook = algorithms.eaMuCommaLambda(
            popul, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats, hof, False)

    avgs_list.append(logbook.select("avg"))
    stds_list.append(logbook.select("std"))
    root_hof.update(hof.items)


# Print the individuals with the best evaluation
print("*** Hall of fame *** ")
for idx, indiv in enumerate(root_hof.items):
    print(f"{idx + 1}. {indiv} = {np.sum(indiv * p)}")


# Plot the generations through the evolutions
x = range(ngen + 1)
avg = np.average(np.array(avgs_list), axis=0)
std = np.sqrt(np.average(np.array(stds_list) ** 2, axis=0))
plt.title('Best evaluation curve')
plt.plot(x, avg, color='red', marker='*')
plt.plot(x, avg + std, color='b', linestyle='-.')
plt.plot(x, avg - std, color='b', linestyle='-.')
plt.xlabel('Generations')
plt.ylabel(f'Best profit found with weight <= {wmax}')
plt.legend(['average', 'average + std', 'average - std'])
plt.savefig(f'{sys.argv[1]}.png')
