import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        converged = False
        current_iter = 0
        while (not converged) and (current_iter < self.max_iter):
            cluster_list = [[] for i in range(self.n_clusters)]
            for x in X:
                distances_list = []
                for c in self.centroids:
                    distances_list.append(np.linalg.norm(x - c))
                cluster_index = np.argmin(distances_list)
                cluster_list[cluster_index].append(x)
            cluster_list = list((filter(None, cluster_list)))
            prev_centroids = self.centroids.copy()
            for i in range(len(cluster_list)):
                self.centroids[i] = np.average(cluster_list[i], axis=0)
            converged = (set([tuple(a) for a in self.centroids]) == set([tuple(a) for a in prev_centroids]))
            current_iter += 1

    def predict(self, X):
        labels = []
        for x in X:
            distances_list = []
            for c in self.centroids:
                distances_list.append(np.linalg.norm(x - c))
            cluster_index = np.argmin(distances_list)
            labels.append(cluster_index)
        return labels


