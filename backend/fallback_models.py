"""
Fallback models for testing and environments where native DLLs are blocked.
"""
import numpy as np

class FallbackScaler:
    """Mimics sklearn.preprocessing.StandardScaler interface."""

    def __init__(self, n_features=15):
        self.mean_ = np.zeros(n_features)
        self.scale_ = np.ones(n_features)
        self.n_features_in_ = n_features

    def transform(self, X):
        return (np.array(X) - self.mean_) / self.scale_

    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self.transform(X)

class FallbackGMM:
    """Mimics sklearn.mixture.GaussianMixture interface with random assignment."""

    def __init__(self, n_components=8):
        self.n_components = n_components
        np.random.seed(42)
        self.means_ = np.random.rand(n_components, 15)

    def predict(self, X):
        X = np.array(X)
        distances = np.array([
            np.sum((X - center) ** 2, axis=1)
            for center in self.means_
        ]).T
        return np.argmin(distances, axis=1)

    def predict_proba(self, X):
        X = np.array(X)
        distances = np.array([
            np.sum((X - center) ** 2, axis=1)
            for center in self.means_
        ]).T
        exp_neg_dist = np.exp(-distances)
        probs = exp_neg_dist / exp_neg_dist.sum(axis=1, keepdims=True)
        return probs

class FallbackXGBRegressor:
    """Mimics xgboost.XGBRegressor interface with a simple heuristic."""

    def predict(self, X):
        X = np.array(X)
        if X.shape[1] >= 15:
            scores = (
                0.3 * (1 - np.clip(X[:, 10], 0, 1))
                + 0.2 * np.clip(X[:, 11], 0, 1)
                + 0.2 * np.clip(X[:, 14], 0, 1)
                + 0.15 * (1 - np.clip(X[:, 4], 0, 1))
                + 0.15 * np.clip(X[:, 2], 0, 1)
            )
        else:
            scores = np.full(X.shape[0], 0.5)
        return np.clip(scores, 0, 1)
