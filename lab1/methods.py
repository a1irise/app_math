from utils import *


def dichotomous_search(a, b, epsilon):
    print_header()

    iteration = 0
    function_calls = 0
    prev_interval_size = b - a

    print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    while 0.5 * (b - a) > epsilon:
        iteration += 1
        function_calls += 2
        prev_interval_size = b - a

        middle = 0.5 * (a + b)
        c = middle - 0.5 * epsilon
        d = middle + 0.5 * epsilon
        f_c = calc_function(c)
        f_d = calc_function(d)

        if f_c < f_d:
            b = d
        else:
            a = c

        print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    print_footer("Dichotomous search", 0.5 * (a + b))


def golden_section_search(a, b, epsilon):
    print_header()

    iteration = 0
    function_calls = 2
    prev_interval_size = b - a

    print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    c = b + tau * (a - b)
    d = a + tau * (b - a)
    f_c = calc_function(c)
    f_d = calc_function(d)

    while b - a > epsilon:
        iteration += 1
        function_calls += 1
        prev_interval_size = b - a

        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = b + tau * (a - b)
            f_c = calc_function(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + tau * (b - a)
            f_d = calc_function(d)

        print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    print_footer("Golden section search", 0.5 * (a + b))


def fibonacci_search(a, b, epsilon):
    print_header()

    iteration = 0
    function_calls = 2
    prev_interval_size = b - a

    print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    n = 1
    while (b - a) / fibonacci(n) >= epsilon:
        n += 1

    c = a + fibonacci(n - 2) / fibonacci(n) * (b - a)
    d = a + fibonacci(n - 1) / fibonacci(n) * (b - a)
    f_c = calc_function(c)
    f_d = calc_function(d)

    while n > 2:
        iteration += 1
        function_calls += 1
        prev_interval_size = b - a

        n -= 1

        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = a + fibonacci(n - 2) / fibonacci(n) * (b - a)
            f_c = calc_function(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + fibonacci(n - 1) / fibonacci(n) * (b - a)
            f_d = calc_function(d)

        print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    print_footer("Fibonacci search", 0.5 * (a + b))


def parabolic_interpolation(a, b, epsilon):
    print_header()

    iteration = 0
    function_calls = 3
    prev_interval_size = b - a

    print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))

    x = 0.5 * (a + b)
    u = b + 1

    f_a = calc_function(a)
    f_x = calc_function(x)
    f_b = calc_function(b)

    while True:
        iteration += 1
        function_calls += 1
        prev_interval_size = b - a

        u_prev = u
        p = math.pow(b - x, 2) * (f_x - f_a) + math.pow(a - x, 2) * (f_b - f_x)
        q = (b - x) * (f_x - f_a) + (a - x) * (f_b - f_x)
        u = x + 0.5 * p / q
        f_u = calc_function(u)

        if abs(u - u_prev) < epsilon:
            print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))
            print_footer("Parabolic interpolation", u)
            break

        if x < u:
            if f_x < f_u:
                b = u
                f_b = f_u
            else:
                a = x
                f_a = f_x
                x = u
                f_x = f_u
        else:
            if f_x < f_u:
                a = u
                f_a = f_u
            else:
                b = x
                f_b = f_x
                x = u
                f_x = f_u

        print_iteration(iteration, function_calls, a, b, prev_interval_size / (b - a))


def brents_method(a, b, epsilon):
    print_brent_header()

    iteration = 0
    function_calls = 1
    prev_interval_size = b - a

    print_brent_iteration(iteration, "start", function_calls, a, b, prev_interval_size / (b - a))

    start = a + 0.5 * (3 - math.sqrt(5)) * (b - a)
    f_start = calc_function(start)

    v = w = x = start
    f_v = f_w = f_x = f_start

    while b - a > epsilon:
        iteration += 1
        function_calls += 1
        prev_interval_size = b - a

        p = math.pow(w - x, 2) * (f_x - f_v) + math.pow(v - x, 2) * (f_w - f_x)
        q = (w - x) * (f_x - f_v) + (v - x) * (f_w - f_x)
        if q == 0 or a > x + 0.5 * p / q > b:
            # golden section
            if x >= 0.5 * (a + b):
                u = b + tau * (a - b)
            else:
                u = a + tau * (b - a)
            method = "golden"
        else:
            # parabolic interpolation
            u = x + 0.5 * p / q
            method = "parabolic"

        # update variables
        f_u = calc_function(u)
        if f_u <= f_x:
            if u < x:
                b = x
            else:
                a = x
            v = w
            f_v = f_w
            w = x
            f_w = f_x
            x = u
            f_x = f_u
        else:
            if u < x:
                a = u
            else:
                b = u
            if f_u <= f_w or w == x:
                v = w
                f_v = f_w
                w = u
                f_w = f_u
            elif f_u <= f_v or v == x or v == w:
                v = u
                f_v = f_u

        print_brent_iteration(iteration, method, function_calls, a, b, prev_interval_size / (b - a))

    print_brent_footer("Brent's method", 0.5 * (a + b))
