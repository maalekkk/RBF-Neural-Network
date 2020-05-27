import numpy as np
from copy import deepcopy, copy


def euclidean_distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def nodes(dataset, num_of_nodes):
    nodes_weights = np.ndarray((num_of_nodes, np.size(dataset, 1)))
    temp = int(np.random.random() * len(dataset))
    nodes_weights[0, :] = dataset[temp, :]
    data_tmp = np.append(dataset[:temp, :], dataset[temp + 1:, :], axis=0)
    for i in range(1, num_of_nodes):
        farthest_dist, farthest_idx = 0, 0
        for point in range(len(data_tmp)):
            nearest_dist = np.linalg.norm(data_tmp[point, :] - nodes_weights[0, :])
            for j in range(1, i):
                if np.linalg.norm(data_tmp[point, :] - nodes_weights[j, :]) < nearest_dist:
                    nearest_dist = np.linalg.norm(data_tmp[point, :] - nodes_weights[j, :])
            if nearest_dist > farthest_dist:
                farthest_dist = nearest_dist
                farthest_idx = point
        nodes_weights[i, :] = data_tmp[farthest_idx, :]
        data_tmp = np.append(data_tmp[:farthest_idx, :], data_tmp[farthest_idx + 1:, :], axis=0)
    return nodes_weights


class KMeans:
    def __init__(self, dataset, num_of_nodes):
        self.centroids = nodes(dataset, num_of_nodes)  # list of centroids

    def fit(self, dataset):
        num_of_classes = len(self.centroids)
        clusters = np.zeros(len(dataset))
        # Loop will run till the error becomes zero
        error = 1
        while error > 0.05:
            # Assigning each value to its closest cluster
            for i in range(len(dataset)):
                distances = euclidean_distance(dataset[i], self.centroids)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Storing the old centroid values
            old_centroids = deepcopy(self.centroids)
            # Finding the new centroids by taking the average value
            for i in range(num_of_classes):
                points = [dataset[j] for j in range(len(dataset)) if clusters[j] == i]
                if len(points) != 0:
                    self.centroids[i] = np.mean(points, axis=0)
            # Store actual error = difference in two iterations
            error = euclidean_distance(self.centroids, old_centroids, None)
        return self.centroids
