import numpy as np


def get_distance(x1, x2):
    s = 0
    for i in range(len(x1)):
        s += (x1[i] - x2[i]) ** 2
    return np.sqrt(s)


def k_means(data, num_of_centers):
    centers = data[np.random.choice(range(len(data)), num_of_centers)]
    equals = False
    iter = 0
    while (not equals) and (iter < 1000):
        cluster_list = [[] for i in range(len(centers))]
        for x in data:
            distances_list = []
            for c in centers:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)
        cluster_list = list((filter(None, cluster_list)))
        prev_centers = centers.copy()
        centers = []
        for j in range(len(cluster_list)):
            centers.append(np.mean(cluster_list[j], axis=0))
        equals = (np.abs(np.sum(prev_centers) - np.sum(centers)) == 0)
        iter += 1
    return np.array(centers)
