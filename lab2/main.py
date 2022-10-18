from lab2.methods import *
from lab2.functions import *
import numpy as np


def test(f, grad, x0, epsilon):
    x1 = with_const_lr(f, grad, x0, 0.1, epsilon)
    print(x1)
    print("--------")
    x2 = with_const_lr(f, grad, x0, 0.9, epsilon)
    print(x2)
    print("--------")
    x3 = with_step_crushing(f, grad, x0, 0.9, epsilon)
    print(x3)
    print("--------")
    x4 = with_golden_section(f, grad, x0, epsilon)
    print(x4)
    print("--------")
    x5 = with_fibonacci(f, grad, x0, epsilon)
    print(x5)
    print("--------")
    x6 = fletcher_reeves(f, grad, x0, epsilon)
    print(x6)


test(f1, grad_f1, np.array([10, 10]), 1e-3)
