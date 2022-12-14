import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE 
    
    return -np.sum(np.mean(y, axis=0) * np.log(np.mean(y, axis=0) + EPS))

    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    
    return 1 - (np.mean(y, axis=0)*np.mean(y, axis=0)).sum()
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    
    return np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return (np.mean(np.abs(y - np.median(y))))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE 
        l = X_subset[:, feature_index] < threshold       
        r = X_subset[:, feature_index] >= threshold
      
        return (X_subset[l, :], y_subset[l, :]), (X_subset[r, :], y_subset[r, :])
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_part = X_subset[:, feature_index] < threshold 
        right_part = X_subset[:, feature_index] >= threshold 

        return y_subset[left_part], y_subset[right_part]

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        feature_index = 0
        threshold = 0
        criterion = self.criterion
        min_criterion = np.inf
        
        for i in range(X_subset.shape[1]):
            thresholds = np.sort(np.unique(X_subset[:, i]))
            for a in thresholds:
                y_left, y_right = self.make_split_only_y(i, a, X_subset, y_subset)
                min_criterion_new =  (len(y_left) * criterion(y_left) + len(y_right) * criterion(y_right)) / X_subset.shape[0]
                if min_criterion_new < min_criterion:
                    feature_index = i
                    threshold = a
                    min_criterion = min_criterion_new

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth = 0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE 
        
        y_same = np.zeros_like(y_subset)
        y_same[:, 0] = 1
        eps = 1e-8
        
        if depth >= self.max_depth or len(y_subset) <= self.min_samples_split or \
           self.criterion(y_subset) < self.criterion(y_same) + eps:
            prediction = 0
            if self.classification:
                prediction = np.argmax(np.sum(y_subset, axis=0))
                probas = np.mean(y_subset, axis = 0)
                return Node(0, prediction, probas)
            else:
                if self.criterion_name == 'variance':
                    prediction = np.mean(y_subset)
                else:
                    prediction = np.median(y_subset)
                return Node(0, prediction)
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        left_subsets, right_subsets = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node.left_child = self.make_tree(*left_subsets, depth + 1)
        new_node.right_child = self.make_tree(*right_subsets, depth + 1)
               
        return new_node 
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted = np.ndarray(n_objects)
        for i in range(n_objects):
            node = self.root
            while node.left_child is not None or node.right_child is not None:
                if X[i][node.feature_index] < node.value and node.left_child is not None:
                    node = node.left_child
                elif node.right_child is not None:
                    node = node.right_child
                y_predicted[i] =  node.value
        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted_probs = np.zeros((n_objects, self.n_classes))
        for index, obj in enumerate(X):
            node = self.root
            while node.proba is None:
                if obj[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
            y_predicted_probs[index] = node.proba

        return y_predicted_probs
