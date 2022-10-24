from random import choice
import numpy as np
from scipy.sparse import csr_matrix


def diagonal_prevalence(n):
    matrix = np.zeros((n, n))

    # fill non-diagonal
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = choice([1, 2, 3, 4, 5])

    # fill diagonal
    for i in range(n):
        non_diagonal_sum = 0
        for j in range(n):
            if i != j:
                non_diagonal_sum += matrix[i][j]
        matrix[i][i] = non_diagonal_sum + 10

    return csr_matrix(matrix)


def hilbert(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1 / (i + j + 1)

    return csr_matrix(matrix)


def get_b(matrix):
    n = matrix.shape[0]
    b = np.zeros(n)

    for i in range(n):
        for j in range(n):
            b[i] += matrix[i, j] * (j + 1)

    return b
