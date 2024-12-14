import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
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
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform', verbose=True):

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
    # Calculate distances to nearest neighbors using Minkowski distance
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
        # Save train data
        if isinstance(X_train, pd.DataFrame):
            if X_train.columns.empty:
                self.X_train = X_train.values.astype(float)
            else:
                self.X_train = X_train.iloc[:, :-1].values.astype(float)
        else:
            self.X_train = X_train.astype(float)

        self.y_train = y_train
       
        
    def _predict_instance(self, row):
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
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))
        