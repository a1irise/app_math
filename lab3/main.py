from datetime import datetime

import numpy as np

from lab3.generators import hilbert, diagonal_prevalence, get_b
from lab3.methods import gauss, jacobi

diag10 = diagonal_prevalence(10)
diag50 = diagonal_prevalence(50)
diag100 = diagonal_prevalence(100)

hilbert10 = hilbert(10)
hilbert50 = hilbert(50)
hilbert100 = hilbert(100)


def test_gauss():
    # diagonal_prevalence
    for matrix in [diag10, diag50, diag100]:
        # matrix = diagonal_prevalence(n)
        n = matrix.shape[0]
        b = get_b(matrix)
        x_exact = np.array(range(1, n + 1))

        t0 = datetime.now()
        x = gauss(matrix, b)
        t1 = datetime.now()

        err = sum(abs(x_exact[i] - x[i]) for i in range(n))

        print("diagonal_prevalence, n=" + str(n)
              + ", runtime=" + str((t1 - t0).microseconds) + " microseconds"
              + ", error=" + str(err))

    # hilbert
    for matrix in [hilbert10, hilbert50, hilbert100]:
        # matrix = hilbert(n)
        n = matrix.shape[0]
        b = get_b(matrix)
        x_exact = np.array(range(1, n + 1))

        t0 = datetime.now()
        x = gauss(matrix, b)
        t1 = datetime.now()

        err = sum(abs(x_exact[i] - x[i]) for i in range(n))

        print("hilbert, n=" + str(n)
              + ", runtime=" + str((t1 - t0).microseconds) + " microseconds"
              + ", error=" + str(err))


def test_jacobi():
    # diagonal_prevalence
    for matrix in [diag10, diag50, diag100]:
        for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            # matrix = diagonal_prevalence(n)
            n = matrix.shape[0]
            b = get_b(matrix)
            x_exact = np.array(range(1, n + 1))

            # # test 1
            # x0 = np.zeros(n)
            #
            # t0 = datetime.now()
            # x, it = jacobi(matrix, b, x0=x0, epsilon=eps)
            # t1 = datetime.now()
            #
            # err = sum(abs(x_exact[i] - x[i]) for i in range(n))
            #
            # print("diagonal_prevalence, n=" + str(n)
            #       + ", eps=" + str(eps)
            #       + ", x0=0"
            #       + ", runtime=" + str((t1 - t0).seconds) + " microseconds"
            #       + ", iterations=" + str(it)
            #       + ", error=" + str(err))
            #
            # # test 2
            # x0 = b
            #
            # t0 = datetime.now()
            # x, it = jacobi(matrix, b, x0, eps)
            # t1 = datetime.now()
            #
            # err = sum(abs(x_exact[i] - x[i]) for i in range(n))
            #
            # print("diagonal_prevalence, n=" + str(n)
            #       + ", eps=" + str(eps)
            #       + ", x0=b"
            #       + ", runtime=" + str((t1 - t0).microseconds) + " microseconds"
            #       + ", iterations=" + str(it)
            #       + ", error=" + str(err))

            # test 3
            x0 = [1000] * n

            t0 = datetime.now()
            x, it = jacobi(matrix, b, x0, eps)
            t1 = datetime.now()

            err = sum(abs(x_exact[i] - x[i]) for i in range(n))

            print("diagonal_prevalence, n=" + str(n)
                  + ", eps=" + str(eps)
                  + ", x0=1000"
                  + ", runtime=" + str((t1 - t0).seconds) + " microseconds"
                  + ", iterations=" + str(it)
                  + ", error=" + str(err))


print("===== gauss =====")
test_gauss()
print("===== jacobi =====")
test_jacobi()
