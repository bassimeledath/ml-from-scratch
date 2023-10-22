import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_priors = None
        self.log_likelihoods = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.log_priors = np.zeros(n_classes, dtype=np.float64)
        self.log_likelihoods = np.zeros(
            (n_classes, n_features), dtype=np.float64)

        self._compute_log_priors(y)
        self._compute_log_likelihoods(X, y, n_features)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        log_posteriors = self.log_priors + \
            np.sum(self.log_likelihoods * x, axis=1)
        return self.classes[np.argmax(log_posteriors)]

    def _compute_log_likelihoods(self, X, y, n_features):
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            N_y = np.sum(X_c)
            N_yi = np.sum(X_c, axis=0)
            probs = (N_yi + self.alpha) / (N_y + self.alpha * n_features)
            self.log_likelihoods[i, :] = np.log(probs)

    def _compute_log_priors(self, y):
        for i, c in enumerate(self.classes):
            self.log_priors[i] = np.log(np.sum(y == c) / len(y))
