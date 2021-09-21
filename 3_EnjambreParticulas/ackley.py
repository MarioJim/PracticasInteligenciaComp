import numpy as np


def ackley_fun(v: np.ndarray) -> np.ndarray:
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = v.shape[1]
    term1 = -a * np.exp(-b * np.sqrt(np.square(v).sum(1) / d))
    term2 = -np.exp(np.cos(c * v).sum(1) / d)
    return term1 + term2 + a + np.e


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def generate_graph(lim: float):
        divs = 80
        x = np.linspace(-lim, lim, divs)
        y = np.linspace(-lim, lim, divs)
        X, Y = np.meshgrid(x, y)
        Z = ackley_fun(np.dstack((X, Y)).reshape(-1, 2))
        Z = Z.reshape(X.shape[0], -1)

        plt.figure(figsize=(6, 5), dpi=160)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title(
            'Ackley function in 2D [{}, {}]'.format(-lim, lim))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('ackley(x1, x2)')
        plt.tight_layout()

    generate_graph(32.768)
    plt.savefig('ackley.png')
    generate_graph(1)
    plt.savefig('ackley2.png')
