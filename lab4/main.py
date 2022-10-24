from lab4.generators import diagonal_prevalence, hilbert
from lab4.jacobi import jacobi_rotate


n = 10

matrix = diagonal_prevalence(n)

for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]:
    values, vectors, it = jacobi_rotate(matrix, epsilon=eps)
    print("diagonal_prevalence, n=" + str(n)
          + ", eps=" + str(eps)
          + ", iterations=" + str(it))

print()

matrix = hilbert(n)

for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]:
    values, vectors, it = jacobi_rotate(matrix, epsilon=eps)
    print("hilbert, n=" + str(n)
          + ", eps=" + str(eps)
          + ", iterations=" + str(it))

# a = [
#     [4, -30, 60, -35],
#     [-30, 300, -675, 420],
#     [60, -675, 1620, -1050],
#     [-35, 420, -1050, 700]
# ]
#
# w, v = np.linalg.eig(a)
# print(w)
# print(v)
#
# print()
# a = csr_matrix(a)
#
# w, v = jacobi_rotate(a)
# print(w)
# print(v.toarray())

# [[39.,  3.,  2.,  5.,  5.,  2.,  3.,  2.,  2.,  5.],
#  [ 3., 42.,  3.,  3.,  5.,  5.,  5.,  1.,  4.,  3.],
#  [ 2.,  3., 39.,  1.,  3.,  3.,  2.,  5.,  5.,  5.],
#  [ 5.,  3.,  1., 34.,  2.,  3.,  4.,  1.,  4.,  1.],
#  [ 5.,  5.,  3.,  2., 40.,  5.,  4.,  2.,  1.,  3.],
#  [ 2.,  5.,  3.,  3.,  5., 38.,  1.,  2.,  3.,  4.],
#  [ 3.,  5.,  2.,  4.,  4.,  1., 38.,  1.,  3.,  5.],
#  [ 2.,  1.,  5.,  1.,  2.,  2.,  1., 31.,  3.,  4.],
#  [ 2.,  4.,  5.,  4.,  1.,  3.,  3.,  3., 40.,  5.],
#  [ 5.,  3.,  5.,  1.,  3.,  4.,  5.,  4.,  5., 45.]]
