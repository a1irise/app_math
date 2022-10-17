from methods import *


def calc(a, b, epsilon):
    dichotomous_search(a, b, epsilon)
    golden_section_search(a, b, epsilon)
    fibonacci_search(a, b, epsilon)
    parabolic_interpolation(a, b, epsilon)
    brents_method(a, b, epsilon)


calc(3, 7, 1e-3)
# calc(3, 7, 1e-7)
# calc(-4, 2, 1e-3)
# calc(-6, 6, 1e-3)
# calc(-1, 3, 1e-3)
