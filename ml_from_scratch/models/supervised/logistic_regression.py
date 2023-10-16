import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape[0], X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            lin_reg_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(lin_reg_pred)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict(self, X):
        lin_reg_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(lin_reg_pred)
