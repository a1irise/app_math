import math

import numpy

from lab4.generators import hilbert, diagonal_prevalence
from lab4.jacobi import Jacobi

matrix = [
    [1, math.sqrt(2), 2],
    [math.sqrt(2), 3, math.sqrt(2)],
    [2, math.sqrt(2), 1]
]

mtx = hilbert(3)

w, v = numpy.linalg.eig(mtx)
print(w)

m = Jacobi(mtx).jacobi_rotate()
print(m)

# matr = [[4, -30, 60, -35],
#         [-30, 300, -675, 420],
#         [60, -675, 1620, -1050],
#         [-35, 420, -1050, 700]]
#
# w, v = numpy.linalg.eig(matr)
# print(w)
# print(v)
#
# print("matr:")
# print(matr)
#
# Jacobi(matr).jacobi_rotate()
