import argparse
import itertools
import timeit

from numba import cuda
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import tqdm


@cuda.jit()
def kernel_dist(X_train, centroids, dist_matrix):
    """
    Calculates euclidean distance between row and column in numba global.
    """
    row, col = cuda.grid(2) # from 0 to num_objects - 1, from 0 to NUM_CENTROIDS

    if row < dist_matrix.shape[0]:
        current_sum = 0
        for j in range(X_train.shape[1]):
            current_sum += (X_train[row, j] - centroids[col, j]) ** 2
        dist_matrix[row, col] = current_sum


@cuda.jit()
def kernel_pointwise_dist(centr_1, centr_2, distance_array):
    row = cuda.grid(1)
    if row < distance_array.shape[0]:
        current_sum = 0
        for j in range(centr_1.shape[1]):
            current_sum += (centr_1[row, j] - centr_2[row, j]) ** 2
        distance_array[row] = current_sum

class KMeans():
    def __init__(self, num_clusters, metric="euclidean", epsilon=1e-2, max_iters=1000, parallel=False, random_state=42):
        self.num_clusters = num_clusters
        self.metric = metric
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.parallel = parallel
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _init_centroids(self, X_train):
        self.train_mean = X_train.mean(axis=0)
        self.train_std = X_train.std(axis=0)
        centroids = []
        for b_mean, b_std in zip(self.train_mean, self.train_std):
            dim_centroids = np.random.normal(loc=b_mean, scale=b_std, size=self.num_clusters)
            centroids.append(dim_centroids)
        centroids = np.asarray(centroids).T
        return centroids

    def _parallel_dist(self, X_train, centroids):
        num_objects = X_train.shape[0]
        num_centroids = centroids.shape[0]
        X_train_cuda = cuda.to_device(X_train)
        centroids_cuda = cuda.to_device(centroids)
        distance_matrix = cuda.device_array((num_objects, num_centroids))

        kernel_dist[(BLOCKS_PER_GRID, NUM_CENTROIDS // 2), (NTHREADS, 2)](
            X_train_cuda, centroids_cuda, distance_matrix
        )
        dist = np.sqrt(distance_matrix.copy_to_host())
        return dist

    def _calculate_labels(self, X_train, centroids):
        if not self.parallel or self.metric != "euclidean":
            distances = cdist(X_train, centroids, metric=self.metric)
        else:
            distances = self._parallel_dist(X_train, centroids)
        
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_new_centroids(self, X_train, labels):
        centroids = []
        for label in range(self.num_clusters):
            ids = labels == label
            if ids.sum() > 0:
                cluster_centroid = X_train[ids].mean(axis=0)
                centroids.append(cluster_centroid)
            else:
                new_centroid = []
                for b_mean, b_std in zip(self.train_mean, self.train_std):
                    new_centroid.append(np.random.normal(loc=b_mean, scale=b_std))
                centroids.append(np.asarray(new_centroid))
        return np.asarray(centroids)

    def _calculate_parallel_centroid_dist(self, centr_1, centr_2):
        num_objects = centr_1.shape[0]
        centr_1_cuda = cuda.to_device(centr_1)
        centr_2_cuda = cuda.to_device(centr_2)
        distance_array = cuda.device_array((num_objects,))

        kernel_pointwise_dist[(NUM_CENTROIDS // 2), (2)](
            centr_1_cuda, centr_2_cuda, distance_array
        )
        dist = np.sqrt(distance_array.copy_to_host())
        return dist

    def _calculate_centroid_dist(self, centr_1, centr_2):
        distances = []
        if not self.parallel or self.metric != "euclidean":
            for point1, point2 in zip(centr_1, centr_2):
                d = cdist(point1.reshape(1, -1), point2.reshape(1, -1), metric=self.metric)[0][0]
                distances.append(d)
            distances = np.asarray(distances)
        else:
            distances = self._calculate_parallel_centroid_dist(centr_1, centr_2)
        return distances

    def _calculate_dist_centroids_labels(self, X_train, centroids):
        labels = self._calculate_labels(X_train, centroids)
        centroids_ = self._calculate_new_centroids(X_train, labels)
        dist = self._calculate_centroid_dist(centroids, centroids_)
        return labels, centroids_, dist


    def fit(self, X_train):
        """
        Arguments
        ---------
            X_train     (np.ndarray) : shape (num_objects, dims)
        Returns
        -------
            centroids   (np.ndarray) : shape (self.num_clusters, dims) - centers of clusters
        """
        centroids = self._init_centroids(X_train)
        labels, centroids_, dist = self._calculate_dist_centroids_labels(X_train, centroids)
        counter = 0
        while np.any(dist > self.epsilon) and counter < self.max_iters:
            centroids = centroids_.copy()
            labels, centroids_, dist = self._calculate_dist_centroids_labels(X_train, centroids)
            counter += 1

        self.labels = labels
        self.centroids = centroids_
        self.iterations = counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather data for object similarity")
    parser.add_argument(
        "--input_fname", default="./housec8.txt", help="Input file"
    )
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads")
    parser.add_argument("--num_centroids", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    args = parser.parse_args()

    with open(args.input_fname) as fin:
        data = []
        for line in fin:
            tmp = line.strip().split()
            tmp = list(filter(lambda x: len(x) > 0, tmp))
            tmp = list(map(int, tmp))
            data.append(tmp)
    data = pd.DataFrame(data, columns=["R", "G", "B"])
    for column in data.columns:
        data[column] = data[column].astype(np.float32)

    NUM_CENTROIDS = args.num_centroids[0]
    NTHREADS = args.nthreads
    BLOCKS_PER_GRID = int(np.ceil(data.shape[0] / NTHREADS))
    
    kmeans = KMeans(NUM_CENTROIDS)
    kmeans.fit(data.values)
    print(kmeans.iterations, kmeans.centroids)

    kmeans = KMeans(NUM_CENTROIDS, parallel=True)
    kmeans.fit(data.values)
    print(kmeans.iterations, kmeans.centroids)

    timedata = []
    for parallel in [False, True]:
        for NUM_CENTROIDS in tqdm.tqdm(args.num_centroids, desc=f"Parallel {parallel}"):
            kmeans = KMeans(NUM_CENTROIDS, parallel=parallel)
            if parallel:
                time = timeit.timeit(stmt=f"kmeans.fit(data.values)", globals=globals(), number=1)
            time = timeit.timeit(stmt=f"kmeans.fit(data.values)", globals=globals(), number=1)
            timedata.append({
                "paralell": parallel,
                "nthreads": args.nthreads,
                "time": time
            })

    timedata = pd.DataFrame(timedata)
    timedata.to_csv(f"task4_{args.nthreads}.csv", index=False)

