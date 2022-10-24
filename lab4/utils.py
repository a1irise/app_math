import numpy as np
from scipy.sparse import csr_matrix


def multiply(x, y):
    n = x.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += x[i, k] * y[k, j]
    return csr_matrix(result)
