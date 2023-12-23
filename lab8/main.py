import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sympy import Add, solve, Symbol

warnings.filterwarnings("ignore")


class AR3:

    def __init__(self):
        self.a = np.empty(shape=(4, ))
        self.generate_coefs()
        self.eps = np.random.normal(0, 1, 1000)
        self.x = np.empty(shape=(1000,))
        self.x[0] = 0
        self.x[1] = 0
        self.x[2] = 0

    def generate_coefs(self):
        while True:
            self.a[0] = 0
            self.a[1] = (np.random.random(1))[0]
            self.a[2] = (np.random.random(1))[0]
            self.a[3] = (np.random.random(1))[0]
            if self.check_stationary():
                break

    def check_stationary(self):
        z = Symbol("z")
        equation = 1 - self.a[1] * z - self.a[2] * (z ** 2) - self.a[3] * (z ** 3)
        roots = solve(equation, z)
        for root in roots:
            if isinstance(root, Add):
                root = complex(root)
                if np.sqrt(root.real ** 2 + root.imag ** 2) < 1:
                    return False
            else:
                root = float(root)
                if np.abs(root) < 1:
                    return False
        return True

    def generate_first_1000(self):
        for i in range(3, 1000):
            self.x[i] = self.a[0] \
                        + self.a[1] * self.x[i - 1] \
                        + self.a[2] * self.x[i - 2] \
                        + self.a[3] * self.x[i - 3] \
                        + self.eps[i]

    def create_delay_vectors(self):
        self.vectors = np.empty(shape=(997, 3))
        for i in range(2, 999):
            self.vectors[i - 2] = np.array([self.x[i - 2], self.x[i - 1], self.x[i]])
        self.target = self.x[3:].copy()

    def split_delay_vectors(self):
        self.train_vectors = self.vectors[:800].copy()
        self.test_vectors = self.vectors[800:].copy()
        self.train_target = self.target[:800].copy()
        self.test_target = self.target[800:].copy()

    def predict(self, kernel, **kwargs):
        sc_X = StandardScaler()
        sc_y = StandardScaler()

        train_vectors = sc_X.fit_transform(self.train_vectors)
        train_target = sc_y.fit_transform(self.train_target.reshape(-1, 1))
        test_vectors = sc_X.fit_transform(self.test_vectors)
        test_target = sc_y.fit_transform(self.test_target.reshape(-1, 1))

        model = svm.SVR(kernel=kernel, **kwargs)
        model.fit(train_vectors, train_target)

        prediction = model.predict(test_vectors).reshape(-1, 1)
        error = mean_squared_error(test_target, prediction)

        return sc_y.inverse_transform(prediction), error

    def predict_linear(self):
        best_error = 1e32
        best_C = -1
        best_prediction = list()
        for C in np.arange(0.1, 5, 0.1):
            prediction, error = self.predict("linear", C=C)
            if error < best_error:
                best_error = error
                best_C = C
                best_prediction = prediction
        plt.plot(self.test_target, color="c", label="true")
        plt.plot(best_prediction, color="r", label="linear")
        plt.legend(loc="upper right")
        plt.show()
        print(f"Kernel=linear")
        print(f"MSE={best_error}")
        print(f"C={best_C}")

    def predict_poly(self):
        best_error = 1e32
        best_C = -1
        best_degree = -1
        best_coef0 = -1
        best_gamma = -1
        best_prediction = list()
        for C in [0.1, 0.4, 0.7, 1., 1.3]:
            for degree in [2, 3, 4]:
                for coef0 in np.arange(-5, 5, 0.5):
                    for gamma in np.arange(0.1, 1, 0.1):
                        prediction, error = self.predict("poly", C=C, degree=degree, coef0=coef0, gamma=gamma)
                        if error < best_error:
                            best_error = error
                            best_C = C
                            best_degree = degree
                            best_coef0 = coef0
                            best_gamma = gamma
                            best_prediction = prediction
        plt.plot(self.test_target, color="c", label="true")
        plt.plot(best_prediction, color="r", label="poly")
        plt.legend(loc="upper right")
        plt.show()
        print(f"Kernel=poly")
        print(f"MSE={best_error}")
        print(f"C={best_C}")
        print(f"degree={best_degree}")
        print(f"coef0={best_coef0}")
        print(f"gamma={best_gamma}")

    def predict_rbf(self):
        best_error = 1e32
        best_C = -1
        best_gamma = -1
        best_prediction = list()
        for C in np.arange(0.1, 5, 0.1):
            for gamma in np.arange(0.1, 1, 0.1):
                prediction, error = self.predict("rbf", C=C, gamma=gamma)
                if error < best_error:
                    best_error = error
                    best_C = C
                    best_gamma = gamma
                    best_prediction = prediction
        plt.plot(self.test_target, color="c", label="true")
        plt.plot(best_prediction, color="r", label="rbf")
        plt.legend(loc="upper right")
        plt.show()
        print(f"Kernel=rbf")
        print(f"MSE={best_error}")
        print(f"C={best_C}")
        print(f"gamma={best_gamma}")

    def predict_sigmoid(self):
        best_error = 1e32
        best_C = -1
        best_coef0 = -1
        best_gamma = -1
        best_prediction = list()
        for C in np.arange(0.1, 5, 0.1):
            for coef0 in np.arange(-5, 5, 0.5):
                for gamma in np.arange(0.1, 1, 0.1):
                    prediction, error = self.predict("sigmoid", C=C, coef0=coef0, gamma=gamma)
                    if error < best_error:
                        best_error = error
                        best_C = C
                        best_coef0 = coef0
                        best_gamma = gamma
                        best_prediction = prediction
        plt.plot(self.test_target, color="c", label="true")
        plt.plot(best_prediction, color="r", label="sigmoid")
        plt.legend(loc="upper right")
        plt.show()
        print(f"Kernel=sigmoid")
        print(f"MSE={best_error}")
        print(f"C={best_C}")
        print(f"coef0={best_coef0}")
        print(f"gamma={best_gamma}")


ar3 = AR3()
print(ar3.a)
ar3.generate_first_1000()
plt.plot(ar3.x, color="b")
plt.show()
ar3.create_delay_vectors()
ar3.split_delay_vectors()
ar3.predict_linear()
ar3.predict_poly()
ar3.predict_rbf()
ar3.predict_sigmoid()
