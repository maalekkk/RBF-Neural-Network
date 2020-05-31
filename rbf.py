from copy import deepcopy
import numpy as np
from kMeans import k_means, get_distance


class RBF:
    def __init__(self, hidden_neurons):
        self.hidden_neurons = hidden_neurons
        self.centers = None
        self.s = None
        self.weights = np.random.uniform(-1, 1, self.hidden_neurons)

    def _radial_function(self, center, data_point, sigma, coefficient):
        if sigma == 0:
            sigma = 0.7
        beta = coefficient / (2 * (sigma ** 2))
        return np.exp(-np.linalg.norm(center - data_point) ** 2 * beta)

    def calculate_matrix(self, X, coefficient=1):
        matrix = np.zeros((len(X), self.hidden_neurons))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                matrix[data_point_arg, center_arg] = self._radial_function(
                    center, data_point, self.s[center_arg], coefficient)
        return matrix

    def fit_two_stages(self, X, Y, backward_propagation, coefficient=1, epochs=1000, l_rate=0.2, b_rate=0.2):
        self.centers = k_means(X, self.hidden_neurons)
        d_max = np.max([get_distance([c1], [c2]) for c1 in self.centers for c2 in self.centers])
        self.s = np.repeat(d_max / np.sqrt(2 * self.hidden_neurons), self.hidden_neurons)
        error = 0
        if backward_propagation:
            for i in range(epochs):
                previous_weights = deepcopy(self.weights)
                derivative = np.zeros(self.hidden_neurons)
                for j in range(len(X)):
                    output = self.predict(X[j], coefficient)
                    error += (output - Y[j]) ** 2
                    for k in range(len(self.weights)):
                        derivative[k] += (output - Y[j]) * self._radial_function(self.centers[k],
                                                                                 X[j], self.s[k],
                                                                                 coefficient)
                derivative /= len(X)
                error /= len(X) * 2
                if i % 100 == 0:
                    print(error)
                for j in range(len(self.weights)):
                    momentum = self.weights[j] - previous_weights[j]
                    self.weights[j] -= l_rate * derivative[j] + b_rate * momentum
        else:
            matrix = self.calculate_matrix(X)
            self.weights = np.dot(np.linalg.pinv(matrix), Y)
        return error

    def predict(self, data, coefficient=1):
        matrix = self.calculate_matrix(data, coefficient)
        pred = np.dot(matrix, self.weights)
        return pred
