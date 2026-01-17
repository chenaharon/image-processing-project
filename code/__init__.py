"""
Image Processing Project - Style and Motion Recognition
Based on: "Recognizing image 'style' and activities in video using local features and naive Bayes"
by Daniel Keren (2003)
"""

from .feature_extraction import extract_dct_features, extract_spatial_temporal_features, quantize_features
from .naive_bayes_classifier import NaiveBayesClassifier
from .motion_detection import MotionClassifier
from .video_processor import VideoProcessor

__all__ = [
    'extract_dct_features',
    'extract_spatial_temporal_features',
    'quantize_features',
    'NaiveBayesClassifier',
    'MotionClassifier',
    'VideoProcessor'
]



