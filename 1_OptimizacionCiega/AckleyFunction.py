import math


def ackley_fun(x):
    a = 20
    b = 0.2
    c = 2 * math.pi
    d = 2
    term1 = -a * math.exp(-b * math.sqrt((x[0] ** 2 + x[1] ** 2) / d))
    term2 = -math.exp((math.cos(c * x[0]) + math.cos(c * x[1])) / d)
    return term1 + term2 + a + math.e
