import numpy as np
from matplotlib import pyplot as plt


def numeric(initial_state, transition_matrix, epsilon=1e-4):
    std = list()
    current_state = initial_state
    while True:
        previous_state = current_state
        current_state = previous_state @ transition_matrix
        std.append(np.linalg.norm(previous_state - current_state))
        if std[-1] < epsilon:
            return current_state, std


def analytical(transition_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_state = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])
    stationary_state /= np.sum(stationary_state)
    return stationary_state.ravel()


transition_matrix = np.array([
    np.array([0.1, 0.3, 0,   0,   0,   0,   0,   0.6]),
    np.array([0.2, 0.1, 0,   0,   0,   0.7, 0,   0  ]),
    np.array([0,   0.3, 0.1, 0.3, 0,   0.3, 0,   0  ]),
    np.array([0,   0,   0.3, 0.1, 0.6, 0,   0,   0  ]),
    np.array([0,   0,   0,   0.2, 0.1, 0.7, 0,   0  ]),
    np.array([0,   0.3, 0.2, 0,   0,   0.1, 0.4, 0  ]),
    np.array([0.2, 0.3, 0,   0,   0,   0.2, 0.1, 0.2]),
    np.array([0.2, 0,   0,   0,   0,   0,   0.7, 0.1])
])

initial_state_1 = np.array([0.5, 0, 0, 0, 0.1, 0.2, 0, 0.2])
final_state_1, std_1 = numeric(initial_state_1, transition_matrix)
print(f"Initial state: {initial_state_1}")
print(f"Final state after {len(std_1)} iterations: {final_state_1}")
plt.plot(std_1, marker='o', color='b')
plt.show()

print('\n=========================\n')

initial_state_2 = np.array([0.3, 0, 0, 0, 0.7, 0, 0, 0])
final_state_2, std_2 = numeric(initial_state_2, transition_matrix)
print(f"Initial state: {initial_state_2}")
print(f"Final state after {len(std_2)} iterations: {final_state_2}")
plt.plot(std_2, marker='o', color='r')
plt.show()

print('\n=========================\n')

stationary_state = analytical(transition_matrix)
print(f"Analytical solution: {stationary_state}")
