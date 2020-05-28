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

    def fit_two_stages(self, X, Y, backward_propagation, l_rate=0.5):
        self.centers = kmeans(X, self.hidden_shape)
        G = self._calculate_interpolation_matrix(X)
        if backward_propagation:
            self.weights = np.random.uniform(-1, 1, self.hidden_shape)
            error = 2
            while error > 1:
                derivative = np.zeros(self.hidden_shape)
                error = 0
                for i in range(len(X)):
                    output = self.predict(X[i])
                    error += (output - Y[i]) ** 2
                    for weight_idx, weight in enumerate(self.weights):
                        derivative[weight_idx] += (output - Y[i]) * self._kernel_function(self.centers[weight_idx], X[i])
                        # if weight_idx == 0:
                        #     print(derivative[weight_idx])
                derivative /= len(X)
                error /= len(X) * 2
                print(error)
                for i in range(len(self.weights)):
                    self.weights[i] -= l_rate * derivative[i]
                    # if i == 0:
                    #     print(self.weights[i])
                # for weight_idx, weight in enumerate(self.weights):
                #     weight -= l_rate * derivative[weight_idx]
                #     # if weight_idx == 0:
                #     #     print(weight)
        else:
            self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions

    def fit_one_stages(self, approx_1_train_x, approx_1_train_y):
        pass
