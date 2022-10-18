from numpy.linalg import norm
from lab2.utils import *

max_iterations = 100


def with_const_lr(f, grad, x0, learning_rate, epsilon):
    iterations = 0
    xk = np.array([x0])
    for i in range(max_iterations):
        iterations += 1
        step = learning_rate * grad(xk[-1])
        if np.all(np.abs(step) < epsilon):
            break
        next_x = xk[-1] - step
        xk = np.append(xk, np.array([next_x]), axis=0)

    visualize(xk, f)
    return xk, iterations


def with_step_crushing(f, grad, x0, learning_rate, epsilon):
    iterations = 0
    xk = np.array([x0])
    for i in range(max_iterations):
        iterations += 1
        step = learning_rate * grad(xk[-1])
        if np.all(np.abs(step) < epsilon):
            break
        next_x = xk[-1] - step
        if f(xk[-1]) < f(next_x):
            learning_rate *= 0.5
            continue
        xk = np.append(xk, np.array([next_x]), axis=0)

    visualize(xk, f)
    return xk, len(xk), iterations


def with_golden_section(f, grad, x0, epsilon):
    iterations = 0
    xk = np.array([x0])
    for i in range(max_iterations):
        iterations += 1
        learning_rate = golden_section_search(0, 1e6, epsilon, lambda lr: f(xk[-1] - lr * grad(xk[-1])))
        step = learning_rate * grad(xk[-1])
        if np.all(np.abs(step) < epsilon):
            break
        next_x = xk[-1] - step
        xk = np.append(xk, np.array([next_x]), axis=0)

    visualize(xk, f)
    return xk, iterations


def with_fibonacci(f, grad, x0, epsilon):
    iterations = 0
    xk = np.array([x0])
    for i in range(max_iterations):
        iterations += 1
        learning_rate = fibonacci_search(0, 1e6, epsilon, lambda lr: f(xk[-1] - lr * grad(xk[-1])))
        step = learning_rate * grad(xk[-1])
        if np.all(np.abs(step) < epsilon):
            break
        next_x = xk[-1] - step
        xk = np.append(xk, np.array([next_x]), axis=0)

    visualize(xk, f)
    return xk, iterations


def fletcher_reeves(f, grad, x0, epsilon):
    iterations = 0
    xk = np.array([x0])
    dk = -grad(xk[-1])
    while True:
        iterations += 1
        learning_rate = golden_section_search(0, 1e6, epsilon, lambda t: f(xk[-1] + t * dk))
        step = learning_rate * dk
        if np.all(np.abs(step) < epsilon):
            break
        next_x = xk[-1] + step
        xk = np.append(xk, np.array([next_x]), axis=0)
        b = math.pow(norm(grad(xk[-1])), 2) / math.pow(norm(grad(xk[-2])), 2)
        dk = -grad(xk[-1]) + b * dk

    visualize(xk, f)
    return xk, iterations
