import math
import numpy as np
from scipy.sparse import lil_matrix
from lab3.matrix import Matrix


def inverse(x):
    x = Matrix(x)
    return x.inverse()


def multiply(x, y):
    n = len(x)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += x[i, k] * y[k, j]
    return result


class Jacobi:

    def __init__(self, a):
        self.n = len(a)
        self.a = lil_matrix(a, dtype=float)

    def jacobi_rotate(self, epsilon=0.001, max_iterations=1000):
        count = 0
        eigenvalues = self.a
        while True:
            # find the largest element off-diagonal element
            # in the upper triangle of the matrix
            max_element = 0.0
            max_i = 0
            max_j = 1
            for i in range(self.n - 1):
                for j in range(i + 1, self.n):
                    if eigenvalues[i, j] >= max_element:
                        max_element = eigenvalues[i, j]
                        max_i = i
                        max_j = j

            # Check is finished
            if abs(max_element) < epsilon:
                break
            if count > max_iterations:
                break
            count += 1

            # compute rotate angle
            top = 2 * eigenvalues[max_i, max_j]
            bottom = eigenvalues[max_i, max_i] - eigenvalues[max_j, max_j]
            if bottom == 0:
                theta = 0.5 * math.atan(np.inf)
            else:
                theta = 0.5 * math.atan(top / bottom)

            # define rotate matrix
            rot = np.eye(self.n)
            rot[max_i, max_j] = math.sin(theta)
            rot[max_j, max_i] = -math.sin(theta)
            rot[max_i, max_i] = rot[max_j, max_j] = math.cos(theta)

            # compute
            inv_rot = inverse(rot)
            t = multiply(inv_rot, eigenvalues)
            eigenvalues = multiply(t, rot)

        return eigenvalues
