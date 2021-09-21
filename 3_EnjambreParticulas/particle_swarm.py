from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt


class ParticleSwarmOptimization:
    def __init__(self, eval_fun: Callable[[np.ndarray], np.ndarray],
                 eval_range: Tuple[float, float], n_particles: int = 500,
                 n_iterations: int = 100, alpha: float = 2, beta: float = 2,
                 v_max: float = 2, seed: int = None):
        self.eval = eval_fun
        self.eval_range = eval_range
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.v_max = v_max
        self.rng = np.random.default_rng(seed)

    def run(self):
        a, b = self.eval_range
        self.x = (b - a) * self.rng.random((self.n_particles, 2)) + a
        local_best = self.x.copy()
        local_evals = self.eval(local_best)
        self.v = np.zeros((self.n_particles, 2))

        global_best_history = np.ndarray((self.n_iterations, 2))

        for i in range(self.n_iterations):
            # Evaluate points
            evals = self.eval(self.x)
            # Update the global best
            global_best = self.x[evals.argmin()]
            global_best_history[i] = global_best
            # Update the local bests
            local_upd_idxs = np.argmin(np.c_[local_evals, evals], axis=1)
            local_upd_idxs = local_upd_idxs.astype(bool)
            local_upd_mask = np.c_[local_upd_idxs, local_upd_idxs]
            local_best[local_upd_mask] = self.x[local_upd_mask]
            local_evals[local_upd_idxs] = evals[local_upd_idxs]
            # Update velocity with global and local bests
            global_v_weight = self.alpha * self.rng.random(2)
            global_v_term = global_v_weight * (global_best - self.x)
            local_v_weight = self.beta * self.rng.random(2)
            local_v_term = local_v_weight * (local_best - self.x)
            self.v += global_v_term + local_v_term
            # Clip velocities greater than v_max
            v_mag = np.linalg.norm(self.v, axis=1).clip(min=self.v_max)
            self.v = self.v_max * self.v / np.c_[v_mag, v_mag]
            # Update positions
            self.x += self.v

        return global_best_history

    def graph_particle_history(self, particle_history: np.ndarray):
        eval_history = self.eval(particle_history)
        color_progression = np.linspace(1, 0, particle_history.shape[0])

        plt.figure(figsize=(6, 5), dpi=160)
        ax = plt.axes(projection='3d')
        plt.set_cmap('seismic')
        ax.scatter3D(particle_history[:, 0], particle_history[:, 1],
                     eval_history, c=color_progression)
        ax.set_title('Best particle history')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('ackley(x1, x2)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plt.tight_layout()

    def graph_evaluation_history(self, particle_history: np.ndarray):
        history_size = particle_history.shape[0]
        eval_history = self.eval(particle_history)
        monotonic_eval_history = np.minimum.accumulate(eval_history)

        plt.figure(dpi=160)
        plt.set_cmap('seismic')
        plt.title('Best evaluation curve')
        plt.plot(np.arange(history_size), eval_history, marker='.', ls='')
        plt.plot(np.arange(history_size), monotonic_eval_history,
                 color='green', marker=None)
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Best evaluation')
        plt.legend(['Evaluation history through iterations',
                   'Best evaluation through iterations'])


if __name__ == "__main__":
    from ackley import ackley_fun

    pso = ParticleSwarmOptimization(ackley_fun, (-32.768, 32.768), seed=0)
    best_particle_history = pso.run()
    pso.graph_particle_history(best_particle_history)
    plt.savefig('pso_particle.png')
    pso.graph_evaluation_history(best_particle_history)
    plt.savefig('pso_evaluation.png')
    best_evaluation_history = ackley_fun(best_particle_history)
    best_idx = np.argmin(best_evaluation_history)
    best_particle = best_particle_history[best_idx]
    best_evaluation = best_evaluation_history[best_idx]
    print('Best particle found on iteration {}: [{:.6f}, {:.6f}] with an evaluation of {:.6f}'.format(
        best_idx+1, best_particle[0], best_particle[1], best_evaluation))
