import logging

import numpy as np


class UnPadreVariosHijos:

    def __init__(self, eval_func, iterations=50, nChildren=10):
        self.eval_func = eval_func
        self.iter = iterations
        self.nChildren = nChildren
        self.parent = np.random.uniform(-32.768, 32.768, (2))
        self.best_iter = [self.parent]
        self.evaluations = [self.eval_func(self.parent)]
        self.logger = logging.getLogger()
        self.logger.info('Initializing algorithm with parent {} and {} children'.format(
            self.parent, self.nChildren))

    def evaluate(self, children):
        self.logger.debug('Evaluando hijos: {}'.format(children))
        return list(map(self.eval_func, children))

    def mutation(self):
        children = self.parent + \
            np.random.normal(scale=2, size=(self.nChildren, 2))
        return np.clip(children, -32.768, 32.768)

    def run(self):
        for _ in range(self.iter):
            children = self.mutation()
            self.logger.debug('Children are {}'.format(children))
            evaluations = self.evaluate(children)
            self.logger.debug('Evaluations are {}'.format(evaluations))
            self.parent = children[np.argmin(evaluations)]
            self.best_iter.append(self.parent)
            self.evaluations.append(self.eval_func(self.parent))
            self.logger.info('New parent is {} with evaluation of {}'.format(
                self.parent, self.evaluations[-1]))

        return self.best_iter, self.evaluations


if __name__ == '__main__':
    from AckleyFunction import ackley_fun
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    algoritmo = UnPadreVariosHijos(ackley_fun)
    algoritmo.run()
