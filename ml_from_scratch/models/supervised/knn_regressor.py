import numpy as np


class KnnRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        distances = [self._euclidean_distance(
            x, x_train) for x_train in self.X_train]
        top_k_indices = np.argsort(distances)[:self.k]
        return np.mean([self.y_train[i] for i in top_k_indices])

    def predict(self, X):
        return [self._predict(x) for x in X]
