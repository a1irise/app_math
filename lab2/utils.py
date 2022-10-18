import math
import numpy as np
from matplotlib import pyplot as plt

phi = 0.5 * (1 + math.sqrt(5))
tau = 1 / phi


def golden_section_search(a, b, epsilon, function):
    c = b + tau * (a - b)
    d = a + tau * (b - a)
    f_c = function(c)
    f_d = function(d)

    while b - a > epsilon:
        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = b + tau * (a - b)
            f_c = function(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + tau * (b - a)
            f_d = function(d)

    return 0.5 * (a + b)


def fibonacci(n):
    return (1 / math.sqrt(5)) * (pow((1 + math.sqrt(5)) / 2, n) - pow((1 - math.sqrt(5)) / 2, n))


def fibonacci_search(a, b, epsilon, function):
    n = 1
    while (b - a) / fibonacci(n) >= epsilon:
        n += 1

    c = a + fibonacci(n - 2) / fibonacci(n) * (b - a)
    d = a + fibonacci(n - 1) / fibonacci(n) * (b - a)
    f_c = function(c)
    f_d = function(d)

    while n > 2:
        n -= 1
        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = a + fibonacci(n - 2) / fibonacci(n) * (b - a)
            f_c = function(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + fibonacci(n - 1) / fibonacci(n) * (b - a)
            f_d = function(d)

    return 0.5 * (a + b)


def visualize(xk, f):
    # figure
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')

    x = y = np.arange(-10.0, 10.0, 0.1)
    x, y = np.meshgrid(x, y)
    z = f(np.array([x, y]))
    ax.plot_surface(x, y, z, alpha=0.5)

    x = np.array([item[0] for item in xk])
    y = np.array([item[1] for item in xk])
    z = f(np.array([x, y]))
    ax.plot(x, y, z, color='black', linestyle='dotted', marker='.', markerfacecolor='red', markersize=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.show()

    # contour
    fig2 = plt.figure()
    bx = fig2.add_subplot()

    x = y = np.arange(-10, 10, 0.1)
    x, y = np.meshgrid(x, y)
    z = f(np.array([x, y]))
    bx.contourf(x, y, z)

    x = np.array([item[0] for item in xk])
    y = np.array([item[1] for item in xk])
    bx.plot(x, y, color='black', linestyle='dotted', marker='.', markerfacecolor='red', markersize=10)

    bx.set_xlabel('x')
    bx.set_ylabel('y')

    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.show()
