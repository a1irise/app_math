import numpy as np
from scipy.sparse import csr_matrix

max_iterations = 1000


def calc_lu(matrix):
    n = matrix.shape[0]
    l = np.eye(n)
    u = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                s = sum(l[i][k] * u[k][j] for k in range(i))
                u[i][j] = matrix[i, j] - s
            else:
                s = sum(l[i][k] * u[k][j] for k in range(j))
                l[i][j] = (matrix[i, j] - s) / u[j][j]

    return csr_matrix(l), csr_matrix(u)


def gauss(matrix, b):
    n = matrix.shape[0]
    l, u = calc_lu(matrix)

    y = np.zeros(n)
    for i in range(n):
        s = sum(l[i, k] * y[k] for k in range(i))
        y[i] = b[i] - s

    x = np.zeros(n)
    for i in reversed(range(n)):
        s = sum(u[i, k] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / u[i, i]

    return x


def jacobi(matrix, b, x0, epsilon):
    n = matrix.shape[0]
    xk = np.array([x0])

    iterations = 0

    while True:
        iterations += 1

        x_prev = xk[-1]
        x_cur = np.zeros(n)
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += matrix[i, j] * x_prev[j]
            x_cur[i] = (b[i] - s) / matrix[i, i]

        if np.all(np.abs(x_cur - x_prev) < epsilon):
            break
        if iterations > max_iterations:
            break

        xk = np.append(xk, np.array([x_cur]), axis=0)

    return xk[-1], iterations


def inverse(matrix):
    n = matrix.shape[0]
    e = np.eye(n)
    x = np.array([gauss(matrix, e[i]) for i in range(n)])
    return np.transpose(x)
