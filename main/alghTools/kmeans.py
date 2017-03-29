import numpy as np

def km(data, k, randCZ=False):
    """алгоритм kmeans"""

    clusters = [[] for i in range(k)]
    idx_clusters = [[] for i in range(k)]

    centroids = np.array([])

    if randCZ:
        for cz_i in range(k):
            while True:
                cz = data[np.random.randint(len(data))]
                if cz not in centroids:
                    centroids = np.append(centroids, cz)
                    break
    else:
        unD = np.unique(data)
        for i in range(k):
            centroids = np.append(centroids, unD[i])

    def average(clusters):
        c_array = np.array([])
        for c in clusters:
            c_array = np.append(c_array, np.mean(c))
        return c_array

    itr = 1
    while True:
        for it, i in enumerate(data):
            evk_array = np.sqrt((i - centroids) ** 2)
            minIDX = np.argmin(evk_array)
            clusters[minIDX].append(i)
            idx_clusters[minIDX].append(it)

        oldCentroids = centroids.copy()
        centroids = average(np.array(clusters))

        if not np.array_equal(centroids, oldCentroids):
            clusters = [[] for i in range(k)]
            idx_clusters = [[] for i in range(k)]
        else:
            idx_clusters = np.array(idx_clusters)
            break
        itr += 1

    idxClusters_sort = [[] for i in range(k)]
    for i in range(len(centroids)):
        remove_index = np.argmin(centroids)
        # remove_index = np.argmax(centroids)

        idxClusters_sort[i] = idx_clusters[remove_index]
        idx_clusters = np.delete(idx_clusters, remove_index, 0)
        centroids = np.delete(centroids, remove_index)

    """
    for k, idxs in enumerate(idxClusters_sort):
        k_data = data[idxs]
        print('k=%i [%f;%f] mean: %f' % (k+1, min(k_data), max(k_data), np.mean(k_data)))
    print('---------------------\n')
    """

    alpha = np.min(data[idxClusters_sort[-1]])
    parsed_data = data[np.where(data > alpha)]
    return np.array(idxClusters_sort), parsed_data, alpha

