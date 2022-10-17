import math

phi = 0.5 * (1 + math.sqrt(5))
tau = 1 / phi


def calc_function(x):
    return math.sin(x) * pow(x, 3)


# def calc_function(x):
#     return pow(x, 4) + 5 * pow(x, 3) - 10 * x


# def calc_function(x):
#     return 5 * math.sin(2 * x) + pow(x, 2)


# def calc_function(x):
#     return 2 * pow(x, 6) - 13 * pow(x, 5) + 26 * pow(x, 4) - 7 * pow(x, 3) - 28 * pow(x, 2) + 20 * x


def fibonacci(n):
    return (1 / math.sqrt(5)) * (pow((1 + math.sqrt(5)) / 2, n) - pow((1 - math.sqrt(5)) / 2, n))


def print_header():
    print(110 * "-")
    print("[{:^11}]    [{:^16}]    [{:^19}]    [{:^19}]    [{:^19}]"
          .format("iteration", "function_calls", "a", "b", "interval_decrease"))
    print(110 * "-")


def print_iteration(iteration_counter, function_calls, a, b, interval_decrease):
    print("{:^13}    {:^18}    {:^21.9f}    {:^21.9f}    {:^21.9f}"
          .format(iteration_counter, function_calls, a, b, interval_decrease))


def print_footer(method_name, result):
    print(110 * "-")
    print("{}: {:.9f}".format(method_name, result))
    print(110 * "-")
    print()


def print_brent_header():
    print(136 * "-")
    print("[{:^11}]    [{:^20}]    [{:^16}]    [{:^19}]    [{:^19}]    [{:^19}]"
          .format("iteration", "method", "function_calls", "a", "b", "interval_decrease"))
    print(136 * "-")


def print_brent_iteration(iteration_counter, method, function_calls, a, b, interval_decrease):
    print("{:^13}    {:^22}    {:^18}    {:^21.9f}    {:^21.9f}    {:^21.9f}"
          .format(iteration_counter, method, function_calls, a, b, interval_decrease))


def print_brent_footer(method_name, result):
    print(136 * "-")
    print("{}: {:.9f}".format(method_name, result))
    print(136 * "-")
    print()
