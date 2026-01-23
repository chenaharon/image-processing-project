"""
Video Processing Module
Handles video loading, processing, and motion analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .motion_detection import MotionClassifier
    from .feature_extraction import extract_spatial_temporal_features, quantize_features
    from .naive_bayes_classifier import NaiveBayesClassifier
except ImportError:
    from motion_detection import MotionClassifier
    from feature_extraction import extract_spatial_temporal_features, quantize_features
    from naive_bayes_classifier import NaiveBayesClassifier


class VideoProcessor:
    """
    Main class for processing videos and detecting motion.
    """
    
    def __init__(self, block_size: int = 5, num_coefficients: int = 10, num_bins: int = 32):
        # Default values match paper: 10 coefficients, 32 bins
        """
        Initialize video processor.
        
        Args:
            block_size: Size of blocks for feature extraction (default: 5 as in paper)
            num_coefficients: Number of DCT coefficients per block
            num_bins: Number of bins for feature quantization
        """
        self.block_size = block_size
        self.num_coefficients = num_coefficients
        self.num_bins = num_bins
        self.motion_classifier = MotionClassifier()
        self.style_classifier = None
    
    def load_video(self, video_path: str, max_frames: Optional[int] = None, 
                   start_from_center: bool = False) -> List[np.ndarray]:
        """
        Load video frames from file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None for all)
            start_from_center: If True, extract frames from center. If False, extract from start.
        
        Returns:
            List of video frames
        """
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Read all frames first
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        cap.release()
        
        if not all_frames:
            return []
        
        # If max_frames is specified, extract subset
        if max_frames and len(all_frames) > max_frames:
            if start_from_center:
                # Extract from center
                start_idx = (len(all_frames) - max_frames) // 2
                frames = all_frames[start_idx:start_idx + max_frames]
            else:
                # Extract from start (original behavior)
                frames = all_frames[:max_frames]
        else:
            frames = all_frames
        
        return frames
    
    def process_video_motion(self, video_path: str, output_path: Optional[str] = None,
                            max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process video to detect and classify motion types.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            max_frames: Maximum number of frames to process
        
        Returns:
            Tuple of (original_frames, motion_visualized_frames)
        """
        frames = self.load_video(video_path, max_frames)
        visualized_frames = []
        
        prev_frame = None
        for i, frame in enumerate(tqdm(frames, desc="Processing motion")):
            if prev_frame is not None:
                # Compute optical flow
                flow = self.motion_classifier.compute_optical_flow(prev_frame, frame)
                
                # Classify motion types
                motion_map = self.motion_classifier.classify_motion_type(flow, block_size=32)
                
                # Visualize motion
                visualized = self.motion_classifier.visualize_motion(frame, motion_map, block_size=32)
                visualized_frames.append(visualized)
            else:
                visualized_frames.append(frame)
            
            prev_frame = frame
        
        # Save output video if path provided
        if output_path:
            self.save_video(visualized_frames, output_path)
        
        return frames, visualized_frames
    
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """
        Save frames as video file.
        
        Args:
            frames: List of frames to save
            output_path: Output video path
            fps: Frames per second
        """
        if len(frames) == 0:
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in tqdm(frames, desc="Saving video"):
            out.write(frame)
        
        out.release()
    
    def extract_features_for_training(self, video_path: str, 
                                     label: int,
                                     max_frames: Optional[int] = None,
                                     start_from_center: bool = False,
                                     min_activity: Optional[float] = None,
                                     return_activities: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract features from video for training using 5x5x5 spatio-temporal neighborhoods.
        Based on Keren (2003) paper methodology.
        
        Args:
            video_path: Path to video file
            label: Class label for this video
            max_frames: Maximum frames to process
            start_from_center: If True, extract frames from center. If False, extract from start.
            min_activity: Minimum temporal activity threshold for block filtering (if None, no filtering)
            return_activities: If True, also return activity array for each block
        
        Returns:
            Tuple of (features, valid_mask, activities)
            - features: Extracted features (N, num_features) - one feature vector per 5x5x5 neighborhood
            - valid_mask: Boolean array indicating which blocks passed activity filter
            - activities: Optional array of block temporal activities (only if return_activities=True)
        """
        frames = self.load_video(video_path, max_frames, start_from_center=start_from_center)
        
        # Extract spatio-temporal features using 5x5x5 neighborhoods (as in paper)
        features, valid_mask, activities = extract_spatial_temporal_features(
            frames, 
            block_size=self.block_size,  # 5x5 spatial blocks
            num_coefficients=self.num_coefficients,
            temporal_window=5,  # 5 frames temporal window
            min_activity=min_activity,
            return_activities=return_activities
        )
        
        # Features are already in shape (N, num_coefficients)
        return features, valid_mask, activities
    
    def train_style_classifier(self, video_paths: List[str], labels: List[int]):
        """
        Train style classifier on videos.
        
        Args:
            video_paths: List of video file paths
            labels: List of class labels
        """
        all_features = []
        all_labels = []
        
        for video_path, label in tqdm(zip(video_paths, labels), 
                                     desc="Extracting training features",
                                     total=len(video_paths)):
            features, valid_mask = self.extract_features_for_training(video_path, label, min_variance=None)
            # Use all features for now (variance filtering can be added later)
            all_features.append(features)
            all_labels.extend([label] * len(features))
        
        # Concatenate all features
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        # Quantize features (compute global min/max from training data)
        X_quantized, feature_min, feature_max = quantize_features(X, self.num_bins)
        
        # Store normalization parameters for later use
        self.feature_min = feature_min
        self.feature_max = feature_max
        
        # Train classifier
        num_classes = len(np.unique(labels))
        num_features = X_quantized.shape[1]
        
        self.style_classifier = NaiveBayesClassifier(num_classes, num_features, self.num_bins)
        self.style_classifier.fit(X_quantized, y)
        
        print(f"Trained classifier on {len(X)} samples, {num_classes} classes")

