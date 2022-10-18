import numpy as np


def f1(values):
    x = values[0]
    y = values[1]
    return pow(x, 2) + pow(y, 2)


def grad_f1(values):
    x = values[0]
    y = values[1]
    dx = 2 * x
    dy = 2 * y
    return np.array([dx, dy])


def f2(values):
    x = values[0]
    y = values[1]
    return pow(x, 2) + pow(y, 2) + x * y - 4 * x - 5 * y


def grad_f2(values):
    x = values[0]
    y = values[1]
    dx = 2 * x + x - 4
    dy = 2 * y + y - 5
    return np.array([dx, dy])
