import numpy as np
from kMeans import kmeans


class RBF:
    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-np.linalg.norm(center - data_point) ** 2 / 2 * self.sigma ** 2)

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                    center, data_point)
        return G

    def fit(self, X, Y):
        self.centers = kmeans(X, self.hidden_shape)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
