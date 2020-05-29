import numpy as np


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


def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    converged = False

    current_iter = 0

    while (not converged) and (current_iter < max_iters):

        cluster_list = [[] for i in range(len(centroids))]

        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        converged = (pattern == 0)

        current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]
