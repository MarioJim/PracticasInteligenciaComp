import logging

import numpy as np


class VariosPadresVariosHijosTraslape:

    def __init__(self, eval_func, iterations=50, nChildren=10):
        self.eval_func = eval_func
        self.iter = iterations
        self.nChildren = nChildren
        self.parents = np.random.uniform(-32.768, 32.768, (nChildren, 2))

        self.logger = logging.getLogger()
        self.logger.info('Initializing algorithm with parents {} and {} children'.format(
            self.parents, self.nChildren))

        evaluations = self.evaluate(self.parents)
        self.best_iter = [self.parents[np.argmin(evaluations)]]
        self.evaluations = [evaluations[np.argmin(evaluations)]]

    def evaluate(self, children):
        self.logger.debug('Evaluando hijos: {}'.format(children))
        return list(map(self.eval_func, children))

    def mutation(self):
        def get_random_parent():
            return self.parents[np.random.randint(0, len(self.parents))]

        def mutate(p):
            return p + np.random.normal(scale=2, size=(1, 2))
        childrenList = [mutate(get_random_parent())
                        for _ in range(2 * len(self.parents))]
        children = np.concatenate(childrenList, axis=0)
        return np.clip(children, -32.768, 32.768)

    def run(self):
        for _ in range(self.iter):
            children = self.mutation()
            self.logger.debug('Children are {}'.format(children))
            points = np.concatenate((children, self.parents))
            evaluations = self.evaluate(points)
            self.logger.debug('Evaluations are {}'.format(evaluations))
            sortedIdxs = np.argsort(evaluations)
            self.parents = points[sortedIdxs][0:self.nChildren]
            self.best_iter.append(points[sortedIdxs[0]])
            self.evaluations.append(evaluations[sortedIdxs[0]])
            self.logger.info('New parents are {} with best evaluation of {}'.format(
                self.parents, self.evaluations[-1]))

        return self.best_iter, self.evaluations


if __name__ == '__main__':
    from AckleyFunction import ackley_fun
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    algoritmo = VariosPadresVariosHijosTraslape(ackley_fun)
    algoritmo.run()
