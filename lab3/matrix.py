import numpy as np
from scipy.sparse import lil_matrix


class Matrix:

    def __init__(self, a):
        self.n = len(a)
        self.a = lil_matrix(a, dtype=float)
        self.l = lil_matrix(np.eye(self.n), dtype=float)
        self.u = lil_matrix(np.zeros((self.n, self.n)), dtype=float)
        self.calc_lu()

    def calc_lu(self):
        for i in range(self.n):
            for j in range(self.n):
                if i <= j:
                    s = sum(self.l[i, k] * self.u[k, j] for k in range(i))
                    self.u[i, j] = self.a[i, j] - s
                else:
                    s = sum(self.l[i, k] * self.u[k, j] for k in range(j))
                    self.l[i, j] = (self.a[i, j] - s) / self.u[j, j]

    def gauss(self, b):
        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            s = sum(self.l[i, k] * y[k] for k in range(i))
            y[i] = b[i] - s

        x = np.zeros(self.n, dtype=float)
        for i in reversed(range(self.n)):
            s = sum(self.u[i, k] * x[k] for k in range(i + 1, self.n))
            x[i] = (y[i] - s) / self.u[i, i]

        return x

    def inverse(self):
        e = np.eye(self.n)
        x = np.array([self.gauss(e[i]) for i in range(self.n)])
        return np.transpose(x)

    def jacobi(self, b, epsilon):
        xk = np.array([b])
        while True:
            x_prev = xk[-1]
            x_cur = np.zeros(self.n)
            for k in range(self.n):
                s = sum(self.a[k, i] * x_prev[i] for i in range(self.n) if k != i)
                x_cur[k] = (b[k] - s) / self.a[k, k]
            if np.all(np.abs(x_cur - x_prev) < epsilon):
                break
            xk = np.append(xk, np.array([x_cur]), axis=0)
        return xk[-1]
