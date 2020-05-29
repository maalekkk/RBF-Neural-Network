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


# def get_distance(x1, x2):
#     sum = 0
#     for i in range(len(x1)):
#         sum += (x1[i] - x2[i]) ** 2
#     return np.sqrt(sum)


def kmeans(X, k, random):
    """Performs k-means clustering for 1D input

    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters

    Returns:
        ndarray -- A kx1 array of final cluster centers
    """

    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
    if not random:
        while not converged:
            """
            compute distances for each cluster center to each point 
            where (distances[i, j] represents the distance between the ith point and jth cluster)
            """
            distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

            # find the cluster that's closest to each point
            closestCluster = np.argmin(distances, axis=1)

            # update clusters by taking the mean of all of the points assigned to that cluster
            for i in range(k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    clusters[i] = np.mean(pointsForCluster, axis=0)

            # converge if clusters haven't moved
            converged = np.linalg.norm(clusters - prevClusters) < 1e-6
            prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []

    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds
