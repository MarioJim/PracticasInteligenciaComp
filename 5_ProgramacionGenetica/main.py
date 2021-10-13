from os import name
from deap import base, creator, algorithms, tools, gp
import operator
import random
import numpy as np
import math
from numpy.core.fromnumeric import mean
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.metrics import mean_squared_error

df = pd.DataFrame({
    'x': list(([0,10], [1,9], [2,8], [3,7], [4,6], [5,5], [6,4], [7,3], [8,2], [9,1]), ),
    'f(x)': [90, 82, 74, 66, 58, 50, 42, 34, 26, 18]
})

def eval_func(ind, inputs, target):
    func_eval = toolbox.compile(expr=ind)
    predictions = list(map(func_eval, inputs))
    try:
        return abs(mean_squared_error(target, predictions)),
    except ValueError:
        return 1000

def potencia(x, y):
    try:
        return math.pow(x, y)
    except:
        return 1

def div(a, b):
    try:
        return a / b
    except:
        return 1

pset = gp.PrimitiveSet('MAIN', 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(potencia, 2)

pset.renameArguments(ARG0='x')
pset.addEphemeralConstant('R', lambda: random.randint(-15,15))
pset.addTerminal(math.e, name='e')

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)


toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
toolbox.register('select', tools.selDoubleTournament, fitness_size=2, parsimony_size=1.7, fitness_first=False)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
toolbox.register('evaluate', eval_func, inputs=df['x'].values.tolist(), target=df['f(x)'].values.tolist())
toolbox.register('compile', gp.compile, pset=pset)

toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('max', np.max)
stats.register('mean', np.mean)
stats.register('std', np.std)

hof = tools.HallOfFame(5)
pop = toolbox.population(n=20)

results, log = algorithms.eaSimple(pop, toolbox, cxpb=1.0, mutpb=0.1, ngen=10, stats=stats, halloffame=hof)

for ind in hof:
    print(ind)