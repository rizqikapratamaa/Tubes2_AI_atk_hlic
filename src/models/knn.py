import numpy as np
from collections import Counter

class KNN:
    def __init__(self, n_neighbor=3, metric='euclidean', p=2, weighed=False):
        self.n_neighbor = n_neighbor
        self.metric = metric
        self.p = p
        self.weighed = weighed
        self.X_train = None
        self.Y_train = None
        
        # Precompute distance function based on metric
        self._distance_func = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'minkowski': lambda x1, x2: self._minkowski_distance(x1, x2, self.p)
        }.get(metric, self._euclidean_distance)

    def fit(self, X_train, Y_train):
        # Convert to numpy arrays for faster computations
        self.X_train = np.asarray(X_train)
        self.Y_train = np.asarray(Y_train)

        # Precompute training data characteristics
        self._train_stats()

    def _train_stats(self):
        """Precompute and store training data statistics for faster predictions."""
        # Normalize or standardize data could be added here
        self.train_mean = np.mean(self.X_train, axis=0)
        self.train_std = np.std(self.X_train, axis=0)
        
        # Optional: Compute KD-tree or Ball tree for faster nearest neighbor search
        # This is a simplified version. For real optimization, consider scipy's KDTree
        self._sorted_indices = np.argsort(self.X_train, axis=0)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2, p):
        return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

    def _fast_distances(self, x_test):
        """Compute distances using vectorized operations."""
        # Compute distances between test point and all training points
        diff = self.X_train - x_test
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances

    def _most_common_label(self, labels, distances):
        if self.weighed:
            # Inverse distance weighting
            weights = 1 / (distances + 1e-8)
            weights /= np.sum(weights)
            unique_labels, label_indices = np.unique(labels, return_inverse=True)
            weighted_votes = np.zeros(len(unique_labels))
            
            for i, label_idx in enumerate(label_indices):
                weighted_votes[label_idx] += weights[i]
            
            return unique_labels[np.argmax(weighted_votes)]
        else:
            # Standard voting
            return Counter(labels).most_common(1)[0][0]

    def predict(self, X_test):
        # Convert to numpy array to ensure consistent type
        X_test = np.asarray(X_test)
        
        # Preallocate predictions array
        predictions = np.zeros(len(X_test), dtype=self.Y_train.dtype)
        
        # Vectorized prediction
        for i, x_test in enumerate(X_test):
            # Compute distances
            distances = self._fast_distances(x_test)
            
            # Find k nearest neighbors
            k_indices = np.argpartition(distances, self.n_neighbor)[:self.n_neighbor]
            
            # Get labels and distances of k nearest neighbors
            k_nearest_labels = self.Y_train[k_indices]
            k_nearest_distances = distances[k_indices]
            
            # Predict using voting mechanism
            predictions[i] = self._most_common_label(k_nearest_labels, k_nearest_distances)
        
        return predictions

    def get_params(self):
        return {
            "n_neighbor": self.n_neighbor,
            "metric": self.metric,
            "p": self.p,
            "weighed": self.weighed
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self