"""
Naive Bayes Classifier Module
Based on Keren (2003) paper methodology.
Implements Bernoulli-style Naive Bayes for binary features.
"""

import numpy as np
from typing import Tuple, List, Optional
import pickle


class NaiveBayesClassifier:
    """
    Naive Bayes classifier for discrete features.
    For binary features (num_bins=2), implements Bernoulli-style Naive Bayes.
    Paper methodology: P(C_i) = n_i/n (no smoothing for priors).
    """
    
    def __init__(self, num_classes: int, num_features: int, num_bins: int = 2):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of classes
            num_features: Number of features per sample
            num_bins: Number of bins (2 for binary features)
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_bins = num_bins
        
        # Prior probabilities P(class) - will be set during training
        self.class_priors = np.zeros(num_classes)
        
        # Conditional probabilities P(feature|class)
        # Shape: (num_classes, num_features, num_bins)
        self.feature_probs = np.ones((num_classes, num_features, num_bins))
        
        # Class counts
        self.class_counts = np.zeros(num_classes)
        
        # Smoothing parameter (Laplace smoothing)
        self.alpha = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray, balanced_priors: bool = False):
        """
        Train the classifier.
        
        Paper methodology: P(C_i) = n_i/n (no smoothing for priors).
        
        Args:
            X: Training features (N, num_features) - binary (0/1) or quantized
            y: Training labels (N,)
            balanced_priors: If True, use uniform priors. If False, use P(C_i) = n_i/n
        """
        N = X.shape[0]
        
        # Calculate class priors: P(C_i) = n_i/n (paper methodology)
        for c in range(self.num_classes):
            self.class_counts[c] = np.sum(y == c)
            if balanced_priors:
                self.class_priors[c] = 1.0 / self.num_classes
            else:
                # Paper: P(C_i) = n_i/n (no smoothing)
                self.class_priors[c] = self.class_counts[c] / N if N > 0 else 0
        
        # Calculate conditional probabilities P(feature|class)
        for c in range(self.num_classes):
            class_mask = (y == c)
            class_samples = X[class_mask]
            
            if len(class_samples) == 0:
                continue
            
            for f in range(self.num_features):
                for b in range(self.num_bins):
                    # Count how many samples of class c have feature f = bin b
                    count = np.sum(class_samples[:, f] == b)
                    # Laplace smoothing
                    self.feature_probs[c, f, b] = (count + self.alpha) / (len(class_samples) + self.num_bins * self.alpha)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Test features (N, num_features)
        
        Returns:
            Class probabilities (N, num_classes)
        """
        N = X.shape[0]
        log_probs = np.zeros((N, self.num_classes))
        
        for i in range(N):
            for c in range(self.num_classes):
                # Log probability = log(P(class)) + sum(log(P(feature|class)))
                log_prob = np.log(self.class_priors[c] + 1e-10)
                
                for f in range(self.num_features):
                    bin_value = X[i, f]
                    bin_value = np.clip(int(bin_value), 0, self.num_bins - 1)
                    log_prob += np.log(self.feature_probs[c, f, bin_value] + 1e-10)
                
                log_probs[i, c] = log_prob
        
        # Convert to probabilities using log-sum-exp trick
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_prob)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def save(self, filepath: str):
        """Save classifier to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_classes': self.num_classes,
                'num_features': self.num_features,
                'num_bins': self.num_bins,
                'class_priors': self.class_priors,
                'feature_probs': self.feature_probs,
                'class_counts': self.class_counts,
                'alpha': self.alpha
            }, f)
    
    def load(self, filepath: str):
        """Load classifier from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.num_classes = data['num_classes']
            self.num_features = data['num_features']
            self.num_bins = data['num_bins']
            self.class_priors = data['class_priors']
            self.feature_probs = data['feature_probs']
            self.class_counts = data['class_counts']
            self.alpha = data['alpha']
