from deap import algorithms, base, creator, tools
import numpy as np
import pandas as pd


def eval_fn(i):
    return np.sum(i * p),


def feasible(i):
    return np.sum(i * w) <= 6404180


if __name__ == "__main__":
    p = pd.read_csv('./p08_p.txt', header=None, names=["p"])["p"]
    w = pd.read_csv('./p08_w.txt', header=None, names=["w"])["w"]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Evolution params
    toolbox.register("select", tools.selRoulette)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("mate", tools.cxOnePoint)
    # Evaluation params
    toolbox.register("evaluate", eval_fn)
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0))
    # Population and individual params
    toolbox.register("attribute", np.random.randint, low=0, high=2)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attribute, n=24)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=10)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    hof = tools.HallOfFame(3)

    log = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof,
                              cxpb=1.0, mutpb=1.0, ngen=1000, stats=stats,
                              verbose=True)

    print(hof)
