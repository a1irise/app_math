import datetime
import numpy as np
from lab3.generators import hilbert, diagonal_prevalence, f
from lab3.matrix import Matrix


def test_gauss():
    # diagonal_prevalence
    for n in [10, 50, 100]:
        m = diagonal_prevalence(n)
        b = f(m)
        t0 = datetime.datetime.now()
        matrix = Matrix(m)
        x = matrix.gauss(b)
        t1 = datetime.datetime.now()
        x_exact = np.array(range(1, n + 1))
        err = sum(abs(x_exact[i] - x[i]) for i in range(n))
        print("diagonal_prevalence, n=" + str(n)
              + ", runtime: " + str((t1 - t0).microseconds) + " microseconds"
              + ", error: " + str(err))
    # hilbert
    for n in [10, 50, 100]:
        m = hilbert(n)
        b = f(m)
        t0 = datetime.datetime.now()
        matrix = Matrix(m)
        x = matrix.gauss(b)
        t1 = datetime.datetime.now()
        x_exact = np.array(range(1, n + 1))
        err = sum(abs(x_exact[i] - x[i]) for i in range(n))
        print("hilbert, n=" + str(n)
              + ", runtime: " + str((t1 - t0).microseconds) + " microseconds"
              + ", error: " + str(err))


def test_jacobi():
    # diagonal_prevalence
    for n in [10, 50]:
        for epsilon in [1e-3, 1e-7, 1e-10]:
            m = diagonal_prevalence(n)
            b = f(m)
            t0 = datetime.datetime.now()
            matrix = Matrix(m)
            x = matrix.jacobi(b, epsilon)
            t1 = datetime.datetime.now()
            x_exact = np.array(range(1, n + 1))
            err = sum(abs(x_exact[i] - x[i]) for i in range(n))
            print("diagonal_prevalence, n=" + str(n) + ", eps=" + str(epsilon)
                  + ", runtime: " + str((t1 - t0).microseconds) + " microseconds"
                  + ", error: " + str(err))
    # hilbert
    for n in [10, 50, 100]:
        for epsilon in [1e-3, 1e-7, 1e-10]:
            m = hilbert(n)
            b = f(m)
            t0 = datetime.datetime.now()
            matrix = Matrix(m)
            x = matrix.gauss(b)
            t1 = datetime.datetime.now()
            x_exact = np.array(range(1, n + 1))
            err = sum(abs(x_exact[i] - x[i]) for i in range(n))
            print("hilbert, n=" + str(n) + ", eps=" + str(epsilon)
                  + ", runtime: " + str((t1 - t0).microseconds) + " microseconds"
                  + ", error: " + str(err))


# print("gauss")
# test_gauss()
print("jacobi")
test_jacobi()



# matrix = [
#     [10, 1, -1],
#     [1, 10, -1],
#     [-1, 1, 10]
# ]
#
# m = Matrix(matrix)
# # print(m.l.toarray())
# # print(m.u.toarray())
# print(m.gauss(np.array([11, 10, 10])))
# # print(m.inverse())
# print(m.jacobi(np.array([11, 10, 10]), 0.001))
