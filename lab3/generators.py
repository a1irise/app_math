from random import choice
import numpy as np


def diagonal_prevalence(n):
    ret = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                ret[i, j] = choice([-1, -2, -3, -4])
            else:
                ret[i, j] = choice([0, -1, -2, -3, -4]) / 10
    return ret


def hilbert(n):
    ret = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            ret[i - 1, j - 1] = 1 / (i + j - 1)
    return ret


def f(a):
    n = len(a)
    ret = np.zeros(n)
    for i in range(n):
        for j in range(n):
            ret[i] += a[i, j] * (j + 1)
    return ret
