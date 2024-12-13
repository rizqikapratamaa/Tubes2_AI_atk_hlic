import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

class KNN:
    def __init__(self, n_neighbor=3, metric='euclidean', p=3, weighed=False, n_threads=4):
        self.n_neighbor = n_neighbor
        self.metric = metric
        self.p = p
        self.weighed = weighed
        self.n_threads = n_threads  # Number of threads to use

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(np.array(x1) - np.array(x2)))
    
    def _minkowski_distance(self, x1, x2, p):
        return np.sum(np.abs(np.array(x1) - np.array(x2)) ** p) ** (1 / p)

    def _most_common_label(self, labels, distances):
        if self.weighed:
            weights = [1 / (d + 1e-5) for d in distances]
            weighed_votes = {}
            for label, weight in zip(labels, weights):
                if label in weighed_votes:
                    weighed_votes[label] += weight
                else:
                    weighed_votes[label] = weight
            return max(weighed_votes, key=weighed_votes.get)
        else:
            return Counter(labels).most_common(1)[0][0]

    def _predict_single(self, x):
        """Helper function to predict a single sample."""
        distances = []
        for x_train in self.X_train:
            if self.metric == 'euclidean':
                distance = self._euclidean_distance(x, x_train)
            elif self.metric == 'manhattan':
                distance = self._manhattan_distance(x, x_train)
            elif self.metric == 'minkowski':
                distance = self._minkowski_distance(x, x_train, self.p)
            distances.append(distance)
        
        # Get indices of the k nearest neighbors
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.n_neighbor]

        # Get labels of k nearest neighbors
        k_nearest_label = [self.Y_train[i] for i in k_indices]
        k_nearest_distance = [distances[i] for i in k_indices]

        return self._most_common_label(k_nearest_label, k_nearest_distance)

    def predict(self, X_test):
        """Predict multiple instances in parallel."""
        predictions = []
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Use threads to predict each test sample in parallel
            predictions = list(executor.map(self._predict_single, X_test))
        
        return predictions

    def get_params(self, deep=False):
        return {
            "n_neighbor": self.n_neighbor,
            "metric": self.metric,
            "p": self.p,
            "weighed": self.weighed,
            "n_threads": self.n_threads
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
