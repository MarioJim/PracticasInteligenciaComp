import pandas as pd


if __name__ == "__main__":
    p = pd.read_csv('./p08_p.txt', header=None, names=["p"])
    w = pd.read_csv('./p08_w.txt', header=None, names=["w"])
    table = pd.concat((p, w), axis=1)
    print(table)
