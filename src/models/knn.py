import numpy as np
import concurrent.futures
from tqdm import tqdm
from os import cpu_count
import pickle
import time


class KNN:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):
        if k < 1 or not isinstance(k, int):
            raise ValueError("k must be an integer greater than 0.")
        if metric not in ['manhattan', 'euclidean', 'minkowski']:
            raise ValueError("Invalid metric. Choose from: 'manhattan', 'euclidean', 'minkowski'.")
        if weights not in ['uniform', 'distance']:
            raise ValueError("weights must be either 'uniform' or 'distance'.")
        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("n_jobs must be greater than 0 or -1 to use all available cores.")

        self.k = k
        self.metric = metric
        self.p = p
        self.weights = weights
        self.verbose = verbose
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def _compute_distances(self, test_instance):
        """
        Compute distances between the test instance and all training data points.
        """
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - test_instance) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - test_instance), axis=1)
        elif self.metric == 'minkowski':
            distances = np.sum(np.abs(self.X_train - test_instance) ** self.p, axis=1) ** (1 / self.p)
        else:
            raise ValueError("Unsupported distance metric.")
        return distances

    def _get_k_nearest_neighbors(self, distances):
        """
        Find the k-nearest neighbors and their corresponding indices and weights.
        """
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_distances = distances[k_indices]

        if self.weights == 'distance':
            # Avoid division by zero by replacing zero distances with a small value
            k_weights = 1 / (k_distances + 1e-10)
        else:
            k_weights = np.ones_like(k_distances)

        return k_indices, k_weights

    def fit(self, X_train, y_train):
        """
        Store the training data and labels.
        """
        self.X_train = np.array(X_train, dtype=np.float32)
        self.y_train = np.array(y_train)

    def _predict_instance(self, test_instance):
        """
        Predict the class label for a single test instance.
        """
        distances = self._compute_distances(test_instance)
        k_indices, k_weights = self._get_k_nearest_neighbors(distances)

        k_labels = self.y_train[k_indices]
        if self.weights == 'uniform':
            # Majority voting
            unique_labels, counts = np.unique(k_labels, return_counts=True)
            prediction = unique_labels[np.argmax(counts)]
        else:  # 'distance'
            # Weighted voting
            label_weights = {}
            for label, weight in zip(k_labels, k_weights):
                label_weights[label] = label_weights.get(label, 0) + weight
            prediction = max(label_weights, key=label_weights.get)

        return prediction

    def predict(self, X_test):
        """
        Predict class labels for all test instances in parallel.
        """
        X_test = np.array(X_test, dtype=np.float32)

        if self.verbose:
            print(f"Making predictions using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'}.")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            if self.verbose:
                results = list(tqdm(executor.map(self._predict_instance, X_test), total=len(X_test)))
            else:
                results = list(executor.map(self._predict_instance, X_test))

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Predictions completed in {elapsed_time:.2f} seconds.")

        return np.array(results)

    def save(self, path):
        """
        Save the trained KNN model to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load a trained KNN model from a file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
