import numpy as np
from kMeans import kmeans


class RBF:
    def __init__(self, hidden_shape, random_centroids):
        self.hidden_shape = hidden_shape
        self.centers = None
        self.weights = None
        self.stds = None
        self.random_centroids = random_centroids
        self.w = np.random.randn(self.hidden_shape)

    def _kernel_function(self, center, data_point, sigma, coefficient):
        beta = 2 * (sigma ** 2) * coefficient
        return np.exp(-np.linalg.norm(center - data_point) ** 2 / beta)

    def _calculate_interpolation_matrix(self, X, coefficient):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                    center, data_point, self.stds[center_arg], coefficient)
        return G

    # def _calculate_sigma(self):
    #     dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
    #     self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

    def fit_two_stages(self, X, Y, backward_propagation, coefficient, l_rate=0.001):
        self.centers, self.stds = kmeans(X, self.hidden_shape, self.random_centroids)
        G = self._calculate_interpolation_matrix(X, coefficient)
        if backward_propagation:
            self.weights = np.random.uniform(-1, 1, self.hidden_shape)
            error = 2
            while error > 1:
                derivative = np.zeros(self.hidden_shape)
                error = 0
                for i in range(len(X)):
                    output = self.predict(X[i], coefficient)
                    error += (output - Y[i]) ** 2
                    for weight_idx, weight in enumerate(self.weights):
                        derivative[weight_idx] += (output - Y[i]) * self._kernel_function(self.centers[weight_idx],
                                                                                          X[i], self.stds[weight_idx],
                                                                                          coefficient)
                derivative /= len(X)
                error /= len(X) * 2
                print(error)
                for i in range(len(self.weights)):
                    self.weights[i] -= l_rate * derivative[i]
        else:
            self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X, coefficient):
        G = self._calculate_interpolation_matrix(X, coefficient)
        predictions = np.dot(G, self.weights)
        return predictions

    def fit_one_stages(self, approx_1_train_x, approx_1_train_y):
        pass
