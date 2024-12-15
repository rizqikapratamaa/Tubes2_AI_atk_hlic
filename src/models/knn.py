import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    """
    K-Nearest Neighbors (KNN) Classifier implementation.

    This class implements the KNN algorithm, which can be used for classification tasks.
    It includes functionality to:
    - Choose the distance metric (Euclidean, Manhattan, or Minkowski)
    - Specify the number of neighbors (k)
    - Choose weighting strategy ('uniform' or 'distance') for neighbors
    - Enable parallel prediction using multiple CPU cores for faster computation
    
    Attributes:
        k (int): The number of nearest neighbors to use for classification.
        metric (str): The distance metric to use for finding neighbors.
        p (int or float): The power parameter for Minkowski distance (relevant if metric is Minkowski).
        weights (str): The weighting strategy for neighbors ('uniform' or 'distance').
        n_jobs (int): The number of parallel jobs to use for predictions (-1 uses all CPU cores).
        verbose (bool): Whether to output progress and time logs for predictions.
    """
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):
        """
        Initialize the KNN classifier with the given hyperparameters.
        
        Args:
            k (int): The number of nearest neighbors to consider for classification (default is 5).
            n_jobs (int): Number of parallel jobs for prediction (-1 for all cores, default is 1).
            metric (str): The distance metric ('minkowski', 'euclidean', 'manhattan', default is 'minkowski').
            p (int or float): The power parameter for Minkowski distance (default is 2, for Euclidean).
            weights (str): The weighting strategy for neighbors ('uniform' or 'distance').
            verbose (bool): Whether to display progress and time logs (default is True).
        
        Raises:
            ValueError: If any of the input parameters are invalid (e.g., non-integer k, invalid metric).
        """
        # Validate inputs
        if k < 1:
            raise ValueError("The value of k must be greater than 0.")
        
        if not isinstance(k, int):
            raise ValueError("k must be an integer.")
        
        allowed_metrics = ['manhattan', 'euclidean', 'minkowski']
        if metric not in allowed_metrics:
            raise ValueError(f"Invalid metric. Choose from: {', '.join(allowed_metrics)}.")
        
        if not isinstance(metric, str):
            raise ValueError("metric must be a string.")

        if p < 1:
            raise ValueError("The value of p must be greater than 0.")
        
        if not isinstance(p, (int, float)):
            raise ValueError("p must be a number (integer or float).")
        
        if weights is None:
            self.weights = 'uniform'
        elif weights not in ['uniform', 'distance']:
            raise ValueError("weights must be either 'uniform' or 'distance'.")
        
        if not isinstance(weights, str):
            raise ValueError("weights must be a string.")

        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("n_jobs must be greater than 0 or -1 to use all available cores.")
        
        if not isinstance(n_jobs, int):
            raise ValueError("n_jobs must be an integer.")
        
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean.")
        

        # Set object properties
        self.k = k
        self.verbose = verbose
        self.metric = metric
        self.weights = weights

        # Set p based on the chosen distance metric
        if self.metric == 'manhattan':
            self.p = 1
        elif self.metric == 'euclidean':
            self.p = 2
        else:
            self.p = p

        # Determine the number of CPU cores to use
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs


    def _get_nearest_neighbors(self, test):
        """
        Find the k-nearest neighbors for a given test instance.
        
        This function computes the distance from the test instance to all training instances,
        sorts them, and selects the k nearest neighbors based on the chosen distance metric.
        
        Args:
            test (numpy.ndarray): The test instance (row) for which nearest neighbors are to be found.
        
        Returns:
            tuple: A tuple containing the indices of the nearest neighbors and their corresponding weights.
        """
        distances = np.linalg.norm(self.X_train - test, ord=self.p, axis=1)
        
        weights = None
        if self.weights == 'uniform':
            # For uniform weights, select the k nearest neighbors
            indices = np.argsort(distances)[:self.k]
        elif self.weights == 'distance':
            # For distance-based weights, select the k nearest neighbors
            indices = np.argsort(distances)[:self.k]
            distances = distances[indices]
            
            # Compute weights as the inverse of the distances and normalize them
            weights = 1 / distances
            weights /= np.sum(weights)
        else:
            raise ValueError("Invalid weights. Allowed values are 'uniform' and 'distance'.")
        
        return indices, weights

    
    def fit(self, X_train, y_train):
        """
        Train the KNN classifier using the provided training data.

        Args:
            X_train (numpy.ndarray or pd.DataFrame): Feature matrix for the training data.
            y_train (numpy.ndarray or pd.Series): Labels for the training data.
        
        Raises:
            ValueError: If the input data is not of the correct type.
        """
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train
       
        
    def _predict_instance(self, row):
        """ 
        Make a prediction for a single test instance.
        
        Args:
            row (numpy.ndarray): A single test instance to classify.
        
        Returns:
            int: The predicted class label.
        """
        # Make a prediction for a single instance
        indices, weights = self._get_nearest_neighbors(row)
        
        # Get the labels of the nearest neighbors
        labels = [self.y_train.iloc[neighbour] for neighbour in indices]
        
        if self.weights == 'uniform':
            # For uniform weights, predict the most common label
            prediction = max(set(labels), key=labels.count)
        elif self.weights == 'distance':
            # For distance-based weights, apply the weights to the labels and predict the weighted majority label
            weighted_labels = [weights[i] * labels[i] for i in range(len(labels))]
            prediction = max(set(weighted_labels), key=weighted_labels.count).astype(int)
        else:
            raise ValueError("Invalid weights. Allowed values are 'uniform' and 'distance'.")
        
        return prediction


    def predict(self, X_test):
        """
        Predict the labels for a given test dataset.
        
        Args:
            X_test (numpy.ndarray or pd.DataFrame): The feature matrix of the test dataset.
        
        Returns:
            numpy.ndarray: Predicted labels for all test instances.
        """
        if self.verbose:
            print(f"Making predictions using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'}.")

        # If X_test is a DataFrame, convert it to a NumPy array
        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        # Track prediction time
        start_time = time.time()

        # Use ProcessPoolExecutor for parallel predictions
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
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

        Args:
            path (str): The file path where the model should be saved.
        """
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        """
        Load a trained KNN model from a file.

        Args:
            path (str): The file path from which to load the model.
        
        Returns:
            KNN: The loaded KNN model.
        """
        return pickle.load(open(path, 'rb'))
        