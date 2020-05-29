from copy import deepcopy

import numpy as np
from kMeans import kmeans


class RBF:
    def __init__(self, hidden_neurons, random_centroids):
        self.hidden_neurons = hidden_neurons
        self.centers = None
        self.stds = None
        self.weights = np.random.uniform(-1, 1, self.hidden_neurons)
        self.random_centroids = random_centroids

    def _kernel_function(self, center, data_point, sigma, coefficient):
        beta = coefficient / (2 * (sigma ** 2))
        return np.exp(-np.linalg.norm(center - data_point) ** 2 * beta)

    def _calculate_interpolation_matrix(self, X, coefficient=1):
        G = np.zeros((len(X), self.hidden_neurons))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                    center, data_point, self.stds[center_arg], coefficient)
        return G

    def fit_two_stages(self, X, Y, backward_propagation, coefficient=1, l_rate=0.2, b_rate=0.2):
        # self.centers, self.stds = kmeans(X, self.hidden_neurons, self.random_centroids)
        # self.centers = X[np.random.choice(range(len(X)), self.hidden_neurons, replace=False)]
        self.centers, self.stds = kmeans(X, self.hidden_neurons, max_iters=1000)
        if backward_propagation:
            error = 2
            # while error > 0.3:
            for i in range(1000):
                self.previousWeights = deepcopy(self.weights)
                derivative = np.zeros(self.hidden_neurons)
                error = 0
                for j in range(len(X)):
                    output = self.predict(X[j], coefficient)
                    error += (output - Y[j]) ** 2
                    for k in range(len(self.weights)):
                        derivative[k] += (output - Y[j]) * self._kernel_function(self.centers[k],
                                                                                 X[j], self.stds[k],
                                                                                 coefficient)
                derivative /= len(X)
                error /= len(X) * 2
                if i % 10 == 0:
                    print(error)
                for j in range(len(self.weights)):
                    momentum = self.weights[j] - self.previousWeights[j]
                    self.weights[j] -= l_rate * derivative[j] + b_rate * momentum
        else:
            G = self._calculate_interpolation_matrix(X)
            self.weights = np.dot(np.linalg.pinv(G), Y)
            print(self.weights)

    def predict(self, X, coefficient=1):
        G = self._calculate_interpolation_matrix(X, coefficient)
        predictions = np.dot(G, self.weights)
        return predictions
