"""
Naive Bayes Classifier Module
Based on the paper: "Recognizing image 'style' and activities in video using local features and naive Bayes"
by Daniel Keren (2003)

This module implements a Naive Bayes classifier for image/video style classification.
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict
import pickle


class NaiveBayesClassifier:
    """
    Naive Bayes classifier for discrete features.
    Assumes feature independence (naive assumption).
    """
    
    def __init__(self, num_classes: int, num_features: int, num_bins: int = 32):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of classes to classify
            num_features: Number of features per sample
            num_bins: Number of bins for feature quantization
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_bins = num_bins
        
        # Prior probabilities P(class)
        self.class_priors = np.zeros(num_classes)
        
        # Conditional probabilities P(feature|class)
        # Shape: (num_classes, num_features, num_bins)
        self.feature_probs = np.ones((num_classes, num_features, num_bins))
        
        # Class counts for smoothing
        self.class_counts = np.zeros(num_classes)
        
        # Smoothing parameter (Laplace smoothing)
        self.alpha = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        
        Args:
            X: Training features (N, num_features) - quantized discrete values
            y: Training labels (N,)
        """
        N = X.shape[0]
        
        # Calculate class priors
        for c in range(self.num_classes):
            self.class_counts[c] = np.sum(y == c)
            self.class_priors[c] = (self.class_counts[c] + self.alpha) / (N + self.num_classes * self.alpha)
        
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
        Predict class probabilities for samples.
        
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
                    bin_value = np.clip(bin_value, 0, self.num_bins - 1)
                    log_prob += np.log(self.feature_probs[c, f, bin_value] + 1e-10)
                
                log_probs[i, c] = log_prob
        
        # Convert to probabilities using log-sum-exp trick
        # Subtract max for numerical stability
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_prob)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Test features (N, num_features)
        
        Returns:
            Predicted class labels (N,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def save(self, filepath: str):
        """Save the classifier to a file."""
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
        """Load the classifier from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.num_classes = data['num_classes']
            self.num_features = data['num_features']
            self.num_bins = data['num_bins']
            self.class_priors = data['class_priors']
            self.feature_probs = data['feature_probs']
            self.class_counts = data['class_counts']
            self.alpha = data['alpha']

