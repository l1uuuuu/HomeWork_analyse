from collections import Counter
import numpy as np

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.X, axis=2)
        indices = np.argpartition(distances, self.n_neighbors)[:, :self.n_neighbors]
        neighbors = self.y[indices]
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=neighbors)
