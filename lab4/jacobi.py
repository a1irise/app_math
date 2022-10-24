import math

import numpy as np
from scipy.sparse import csr_matrix

from lab4.utils import multiply

max_iterations = 10000


def jacobi_rotate(matrix, epsilon=0.001):
    iterations = 0
    n = matrix.shape[0]

    eigenvalues = np.zeros(n)
    eigenvectors = csr_matrix(np.eye(n))

    while True:
        iterations += 1

        # find max matrix element by module
        max_element = matrix[0, 1]
        max_i = 0
        max_j = 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(matrix[i, j]) > abs(max_element):
                    max_element = matrix[i, j]
                    max_i = i
                    max_j = j

        # check done
        if abs(max_element) < epsilon:
            break
        if iterations > max_iterations:
            break

        # find rotation angle
        if matrix[max_i, max_i] == matrix[max_j, max_j]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan((2 * matrix[max_i, max_j])
                                    / (matrix[max_i, max_i] - matrix[max_j, max_j]))

        # find rotation matrix
        rotate_matrix = np.eye(n)
        rotate_matrix[max_i][max_j] = -math.sin(theta)
        rotate_matrix[max_j][max_i] = math.sin(theta)
        rotate_matrix[max_i][max_i] = math.cos(theta)
        rotate_matrix[max_j][max_j] = math.cos(theta)
        rotate_matrix = csr_matrix(rotate_matrix)

        # find inverse rotation matrix
        inverse_rotate_matrix = np.eye(n)
        inverse_rotate_matrix[max_i][max_j] = math.sin(theta)
        inverse_rotate_matrix[max_j][max_i] = -math.sin(theta)
        inverse_rotate_matrix[max_i][max_i] = math.cos(theta)
        inverse_rotate_matrix[max_j][max_j] = math.cos(theta)
        inverse_rotate_matrix = csr_matrix(inverse_rotate_matrix)

        # rotate
        temp = multiply(inverse_rotate_matrix, matrix)
        matrix = multiply(temp, rotate_matrix)

        # update eigenvectors
        eigenvectors = multiply(eigenvectors, rotate_matrix)

    for i in range(n):
        eigenvalues[i] = matrix[i, i]

    return eigenvalues, eigenvectors, iterations
