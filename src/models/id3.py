import numpy as np
from collections import Counter

class Node:
    """Node class for decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index for splitting
        self.threshold = threshold  # Threshold value for the split
        self.left = left          # Left subtree
        self.right = right        # Right subtree
        self.value = value        # Leaf node value


class ID3:
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
        self.root = None
        self.class_weights = None
        
    def entropy(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        weights = np.array([self.class_weights.get(c, 1.0) for c in counts.keys()])
        probs = np.array([count/len(y) for count in counts.values()])
        return -np.sum(weights * probs * np.log2(probs + 1e-10))
    
    def information_gain(self, y, X_column, threshold):
        parent_entropy = self.entropy(y)
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l, e_r = self.entropy(y[left_mask]), self.entropy(y[right_mask])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            X_column = X[:, feature]
            # More granular thresholds
            thresholds = np.percentile(X_column, 
                [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
            
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1:
            counts = Counter(y)
            max_count = max(counts.values())
            # Handle ties with class weights
            majority_classes = [c for c, count in counts.items() 
                              if count == max_count]
            if len(majority_classes) > 1:
                weighted_counts = {c: count * self.class_weights[c] 
                                 for c, count in counts.items()}
                leaf_value = max(weighted_counts, key=weighted_counts.get)
            else:
                leaf_value = counts.most_common(1)[0][0]
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # Calculate class weights
        classes, counts = np.unique(y, return_counts=True)
        self.class_weights = dict(zip(classes, len(y)/(len(classes) * counts)))
        
        self.root = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        return self.predict_single(x, node.right)
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self.predict_single(x, self.root) for x in X])
