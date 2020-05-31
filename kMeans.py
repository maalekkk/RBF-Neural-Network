import numpy as np


def get_distance(x1, x2):
    s = 0
    for i in range(len(x1)):
        s += (x1[i] - x2[i]) ** 2
    return np.sqrt(s)


def k_means(data, num_of_centers):
    centroids = data[np.random.choice(range(len(data)), num_of_centers)]
    converged = False
    current_iter = 0
    while (not converged) and (current_iter < 1000):
        cluster_list = [[] for i in range(len(centroids))]
        for x in data:
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
    return np.array(centroids)
