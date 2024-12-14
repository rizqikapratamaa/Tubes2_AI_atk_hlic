import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ID3DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.label_encoder = LabelEncoder()

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -(ps * np.log2(ps)).sum()

    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Vectorized computation for each feature
        for feature_idx in range(n_features):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                if left_mask.sum() < self.min_samples_split or (~left_mask).sum() < self.min_samples_split:
                    continue
                    
                parent_entropy = self.entropy(y)
                left_entropy = self.entropy(y[left_mask])
                right_entropy = self.entropy(y[~left_mask])
                
                n = len(y)
                n_l, n_r = left_mask.sum(), (~left_mask).sum()
                gain = parent_entropy - (n_l/n * left_entropy + n_r/n * right_entropy)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])
            
        left_mask = X[:, feature] <= threshold
        
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        self.root = self.build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.array(X)
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return self.label_encoder.inverse_transform(predictions)
