import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNearestNeighbors:
    """
    KNearestNeighbors is a machine learning algorithm for classification that leverages the concept of finding the k-nearest data points to make predictions. 
    This algorithm stores the entire training dataset in an n-dimensional space and doesn't learn a model upfront. Instead, it performs computations at the time of prediction.

    Upon receiving new data, it searches for the k closest neighbors in the stored dataset and:
    - For classification, it returns the most frequent class among the neighbors.
    - For regression, it returns the average of the values of the nearest neighbors.

    Parameters:
    -----------
    - **n_neighbors** (int, default=5): 
      Number of neighbors to consider for the prediction. This value defines how many neighbors will be analyzed to make a decision.
    
    - **weighting** (str, {'uniform', 'distance'}, default='uniform'): 
      Defines how the algorithm weights the neighbors. 
      - 'uniform' means all neighbors have equal weight.
      - 'distance' gives closer neighbors more influence, weighted by the inverse of their distance.
    
    - **p** (int, default=2): 
      The power parameter for the Minkowski distance metric. When p=1, the metric corresponds to Manhattan distance; p=2 corresponds to Euclidean distance. For other values, Minkowski distance (l_p) is used.

    - **metric** (str, default='minkowski'): 
      The distance metric to compute the distance between data points. By default, the Minkowski metric is used, which defaults to Euclidean distance when p=2.

    - **n_jobs** (int, default=1): 
      The number of CPU cores to use for parallel processing. 
      - Set to -1 to use all available cores. 
      - 1 means a single core will be used.

    - **show_progress** (bool, default=True): 
      A flag to indicate whether progress information should be displayed during execution.

    Methods:
    --------
    - **fit(X_train, y_train)**: 
      This method trains the model using the training data (X_train) and their corresponding labels (y_train).
    
    - **predict(X_test)**: 
      Given new data (X_test), this method predicts the class labels (for classification) or values (for regression) based on the learned data.
    
    - **save_model(path)**: 
      Saves the trained model to the specified file path for later use or sharing.
    
    - **load_model(path)**: 
      Loads a saved model from the given file path for making predictions or further training.

    Example Usage:
    --------------
    ```python
    from src.lib.neighbors import KNearestNeighbors
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize model, train, and predict
    model = KNearestNeighbors(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate accuracy
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    ```
    """

    def __init__(self, n_neighbors=5, n_jobs=-1, metric='minkowski', p=2, weighting='uniform', show_progress=True):

        # Validate input parameters
        if n_neighbors < 1:
            raise ValueError("Invalid n_neighbors. n_neighbors must be greater than 0.")
        
        if not isinstance(n_neighbors, int):
            raise ValueError("Invalid n_neighbors. It must be an integer.")
        
        if metric not in ['manhattan', 'euclidean', 'minkowski']:
            raise ValueError("Invalid metric. Valid metrics are 'manhattan', 'euclidean', and 'minkowski'.")
        
        if not isinstance(metric, str):
            raise ValueError("Invalid metric. It must be a string.")

        if p < 1:
            raise ValueError("Invalid p. p must be greater than 0.")
        
        if not isinstance(p, (int, float)):
            raise ValueError("Invalid p. p must be a number.")
        
        if weighting is None:
            self.weighting = 'uniform'
        elif weighting not in ['uniform', 'distance']:
            raise ValueError("Invalid weighting. Valid values are 'uniform' and 'distance'.")
        
        if not isinstance(weighting, str):
            raise ValueError("Invalid weighting. weighting must be a string.")
        
        if n_jobs < 1 and n_jobs != -1:
            raise ValueError("Invalid n_jobs. n_jobs must be greater than 0 or -1 for all cores.")
        
        if not isinstance(n_jobs, int):
            raise ValueError("Invalid n_jobs. n_jobs must be an integer.")
        
        if not isinstance(show_progress, bool):
            raise ValueError("Invalid show_progress. It must be a boolean.")
    

        # Initialize attributes
        self.n_neighbors = n_neighbors
        self.show_progress = show_progress
        self.metric = metric
        self.weighting = weighting

        # Set the distance power parameter based on the metric
        if self.metric == 'manhattan':
            self.p = 1
        elif self.metric == 'euclidean':
            self.p = 2
        else:
            self.p = p

        # Set the number of parallel jobs
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

    def _find_nearest_neighbors(self, query_point):
        # Compute distances to all training points using the Minkowski distance
        distances = np.linalg.norm(self.X_train - query_point, ord=self.p, axis=1)
        
        weights = None
        if self.weighting == 'uniform':
            # Select the indices of the closest k neighbors
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        elif self.weighting == 'distance':
            # Select the closest k neighbors and compute their weights
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            distances = distances[neighbor_indices]
            weights = 1 / distances
            weights /= np.sum(weights)
        else:
            raise ValueError("Invalid weighting. Valid values are 'uniform' and 'distance'.")
        
        return neighbor_indices, weights
    
    def fit(self, X_train, y_train):
        # Store training data
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train
        
    def _predict_single_instance(self, instance):
        # Predict the label for a single test instance
        neighbor_indices, weights = self._find_nearest_neighbors(instance)
        neighbor_labels = [self.y_train.iloc[idx] for idx in neighbor_indices]
        
        if self.weighting == 'uniform':
            # Majority voting for uniform weighting
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
        elif self.weighting == 'distance':
            # Weighted majority voting for distance weighting
            weighted_labels = [weights[i] * neighbor_labels[i] for i in range(len(neighbor_labels))]
            prediction = max(set(weighted_labels), key=weighted_labels.count).astype(int)
        else:
            raise ValueError("Invalid weighting. Valid values are 'uniform' and 'distance'.")
        
        return prediction

    def predict(self, X_test):
        if self.show_progress:
            print(f"Using {self.n_jobs} {'core' if self.n_jobs == 1 else 'cores'} for predictions.")

        if isinstance(X_test, pd.DataFrame):
            if X_test.columns.empty:
                X_test = X_test.values.astype(float)
            else:
                X_test = X_test.iloc[:, :-1].values.astype(float)
        else:
            X_test = X_test.astype(float)

        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if self.show_progress:
                predictions = list(tqdm(executor.map(self._predict_single_instance, X_test), total=len(X_test)))
            else:
                predictions = list(executor.map(self._predict_single_instance, X_test))

        elapsed_time = time.time() - start_time

        if self.show_progress:
            print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(predictions)
    
    def save_model(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load_model(path):
        return pickle.load(open(path, 'rb'))
