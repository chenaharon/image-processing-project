"""
Feature Extraction Module
Based on the paper: "Recognizing image 'style' and activities in video using local features and naive Bayes"
by Daniel Keren (2003)

This module extracts DCT-based local features from image blocks or video frames.
"""

import numpy as np
from scipy.fftpack import dct
from typing import Tuple, List, Optional
import cv2


def extract_dct_features(image: np.ndarray, block_size: int = 5, num_coefficients: int = 10) -> np.ndarray:
    """
    Extract DCT coefficients from image blocks.
    
    Args:
        image: Input image (grayscale or color)
        block_size: Size of blocks to extract features from (default: 5x5 as in paper)
        num_coefficients: Number of DCT coefficients to use per block (default: 10)
    
    Returns:
        Array of features for each block (N, num_coefficients)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    h, w = gray.shape
    features = []
    
    # Extract features from non-overlapping blocks
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            
            # Apply 2D DCT
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            
            # Extract low-frequency coefficients (zigzag pattern)
            # Start from (0,0) and take first num_coefficients
            coeffs = extract_zigzag_coefficients(dct_block, num_coefficients)
            features.append(coeffs)
    
    return np.array(features)


def extract_zigzag_coefficients(dct_block: np.ndarray, num_coefficients: int) -> np.ndarray:
    """
    Extract DCT coefficients in zigzag pattern (low to high frequency).
    
    Args:
        dct_block: 2D DCT coefficients
        num_coefficients: Number of coefficients to extract
    
    Returns:
        Array of coefficients
    """
    h, w = dct_block.shape
    coeffs = []
    i, j = 0, 0
    direction = 1  # 1 for up-right, -1 for down-left
    
    while len(coeffs) < num_coefficients and (i < h or j < w):
        if 0 <= i < h and 0 <= j < w:
            coeffs.append(dct_block[i, j])
        
        if direction == 1:  # Moving up-right
            if j == w - 1:
                i += 1
                direction = -1
            elif i == 0:
                j += 1
                direction = -1
            else:
                i -= 1
                j += 1
        else:  # Moving down-left
            if i == h - 1:
                j += 1
                direction = 1
            elif j == 0:
                i += 1
                direction = 1
            else:
                i += 1
                j -= 1
    
    # Pad if needed
    while len(coeffs) < num_coefficients:
        coeffs.append(0.0)
    
    return np.array(coeffs[:num_coefficients])


def extract_3d_zigzag_coefficients(dct_3d: np.ndarray, num_coefficients: int) -> np.ndarray:
    """
    Extract DCT coefficients from 3D DCT block in zigzag pattern (low to high frequency).
    Traverses the 3D block starting from (0,0,0) in increasing frequency order.
    
    Args:
        dct_3d: 3D DCT coefficients (shape: depth, height, width, typically 5x5x5)
        num_coefficients: Number of coefficients to extract
    
    Returns:
        Array of coefficients
    """
    d, h, w = dct_3d.shape
    coeffs = []
    
    # Start from DC component (0,0,0)
    # Traverse in order of increasing frequency (sum of indices)
    # Priority: lower sum of (i+j+k) first, then by i, j, k
    visited = set()
    candidates = [(0, 0, 0)]  # Start with DC component
    
    while len(coeffs) < num_coefficients and candidates:
        # Sort by frequency (sum of indices), then by i, j, k
        candidates.sort(key=lambda x: (x[0] + x[1] + x[2], x[0], x[1], x[2]))
        
        i, j, k = candidates.pop(0)
        
        if (i, j, k) in visited:
            continue
        
        if 0 <= i < d and 0 <= j < h and 0 <= k < w:
            coeffs.append(dct_3d[i, j, k])
            visited.add((i, j, k))
            
            # Add neighbors (increasing frequency)
            for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]:
                ni, nj, nk = i + di, j + dj, k + dk
                if (ni, nj, nk) not in visited and ni < d and nj < h and nk < w:
                    candidates.append((ni, nj, nk))
    
    # Pad if needed
    while len(coeffs) < num_coefficients:
        coeffs.append(0.0)
    
    return np.array(coeffs[:num_coefficients])


def compute_temporal_activity(st_volume: np.ndarray) -> float:
    """
    Compute temporal activity of a spatio-temporal block.
    
    Paper: "Blocks with a small time derivative... are not considered" for activity detection.
    We compute the average squared difference between consecutive frames as a measure of temporal activity.
    
    Args:
        st_volume: Spatio-temporal volume (5, 5, 5) - shape: (temporal, spatial_y, spatial_x)
    
    Returns:
        Average squared difference between consecutive frames (temporal activity measure)
    """
    # Compute squared differences between consecutive frames
    # st_volume shape: (5, 5, 5) where first dimension is temporal
    temporal_activity = 0.0
    num_pairs = 0
    
    for t in range(len(st_volume) - 1):
        frame_diff = st_volume[t + 1] - st_volume[t]
        squared_diff = np.mean(frame_diff ** 2)
        temporal_activity += squared_diff
        num_pairs += 1
    
    if num_pairs > 0:
        return temporal_activity / num_pairs
    else:
        return 0.0


def extract_spatial_temporal_features(video_frames: List[np.ndarray], 
                                     block_size: int = 5,
                                     num_coefficients: int = 10,
                                     temporal_window: int = 5,
                                     min_activity: Optional[float] = None,
                                     return_activities: bool = False,
                                     target_resolution: int = 64) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract spatio-temporal features from video frames using 5x5x5 neighborhoods.
    
    Implements the methodology from Keren (2003): "Recognizing image 'style' and 
    activities in video using local features and naive Bayes".
    
    Args:
        video_frames: List of video frames (will be resized to target_resolution)
        block_size: Size of spatial blocks (default: 5x5 as in paper)
        num_coefficients: Number of DCT coefficients per block
        temporal_window: Number of frames in temporal window (default: 5 as in paper)
        min_activity: Minimum temporal activity threshold for block filtering (if None, no filtering)
        return_activities: If True, also return activity array for each block
        target_resolution: Target resolution for resizing frames (default: 64x64 as in paper)
    
    Returns:
        Tuple of (features, valid_block_mask, activities)
        - features: Array of features (N, num_coefficients) - one feature vector per 5x5x5 neighborhood
        - valid_block_mask: Boolean array (N,) indicating which blocks passed activity filter
        - activities: Optional array (N,) of block temporal activities (only if return_activities=True)
    """
    if len(video_frames) < temporal_window:
        raise ValueError(f"Need at least {temporal_window} frames for spatio-temporal features")
    
    # Resize frames to target resolution
    resized_frames = []
    for frame in video_frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        resized = cv2.resize(gray, (target_resolution, target_resolution))
        resized_frames.append(resized)
    
    if len(resized_frames) == 0:
        raise ValueError("No frames extracted from video")
    
    all_features = []
    all_activities = []
    h, w = resized_frames[0].shape
    
    # Extract features from 5x5x5 spatio-temporal neighborhoods
    # Slide through spatial positions
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            # Slide through temporal dimension
            for t in range(len(resized_frames) - temporal_window + 1):
                # Extract 5x5x5 spatio-temporal block
                spatio_temporal_block = []
                for frame_idx in range(t, t + temporal_window):
                    frame = resized_frames[frame_idx]
                    spatial_block = frame[i:i+block_size, j:j+block_size]
                    spatio_temporal_block.append(spatial_block)
                
                # Convert to numpy array: (5, 5, 5) spatio-temporal volume
                st_volume = np.array(spatio_temporal_block)  # Shape: (5, 5, 5)
                
                # Normalize: mean=0, std=1 (Keren 2003 methodology, Section 5)
                # Paper states: "The blocks are first normalized to zero mean and unit variance"
                st_volume = st_volume.astype(np.float32)
                mean_val = np.mean(st_volume)
                std_val = np.std(st_volume)
                if std_val > 1e-6:  # Avoid division by zero
                    st_volume = (st_volume - mean_val) / std_val
                else:
                    st_volume = st_volume - mean_val  # If std=0, just center
                
                # Paper methodology: Apply 3D DCT to the full 5x5x5 spatio-temporal volume
                # This captures both spatial and temporal patterns (as in paper)
                # Apply 3D DCT: first along axis 0 (temporal), then axis 1 (spatial y), then axis 2 (spatial x)
                dct_3d = dct(dct(dct(st_volume, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
                
                # Extract low-frequency coefficients (3D zigzag pattern)
                # Paper uses first num_coefficients (typically ~10, we use 10)
                coeffs = extract_3d_zigzag_coefficients(dct_3d, num_coefficients)
                
                # Compute temporal activity for filtering (paper: "Blocks with a small time derivative... are not considered")
                block_activity = compute_temporal_activity(st_volume)
                
                all_features.append(coeffs)
                all_activities.append(block_activity)
    
    features_array = np.array(all_features)
    activities_array = np.array(all_activities)
    
    # Filter low-activity blocks if threshold is specified
    # Paper: "Blocks with a small time derivative... are not considered" for activity detection
    if min_activity is not None:
        valid_mask = activities_array >= min_activity
    else:
        valid_mask = np.ones(len(features_array), dtype=bool)
    
    if return_activities:
        return features_array, valid_mask, activities_array
    else:
        return features_array, valid_mask, None


def extract_spatial_temporal_features_overlapping(video_frames: List[np.ndarray], 
                                                 block_size: int = 5,
                                                 stride: int = 2,
                                                 num_coefficients: int = 10,
                                                 temporal_window: int = 5,
                                                 min_activity: Optional[float] = None,
                                                 return_activities: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract spatio-temporal features with overlapping blocks (sliding window).
    
    This is an improved version that uses overlapping blocks (stride < block_size)
    instead of non-overlapping blocks. This provides better coverage and smoother predictions.
    
    Args:
        video_frames: List of video frames (should be resized to 64x64)
        block_size: Size of spatial blocks (default: 5x5)
        stride: Stride for sliding window (default: 2 for heavy overlap)
        num_coefficients: Number of DCT coefficients per block
        temporal_window: Number of frames in temporal window (default: 5)
        min_activity: Minimum temporal activity threshold for block filtering
        return_activities: If True, also return activity array for each block
    
    Returns:
        Tuple of (features, valid_block_mask, activities)
    """
    if len(video_frames) < temporal_window:
        raise ValueError(f"Need at least {temporal_window} frames for spatio-temporal features")
    
    # Resize frames to 64x64
    resized_frames = []
    for frame in video_frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        resized = cv2.resize(gray, (64, 64))
        resized_frames.append(resized)
    
    if len(resized_frames) == 0:
        raise ValueError("No frames extracted from video")
    
    all_features = []
    all_activities = []
    h, w = resized_frames[0].shape
    
    # Extract features with overlapping blocks (sliding window)
    # Use stride instead of block_size for sliding
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            # Slide through temporal dimension
            for t in range(len(resized_frames) - temporal_window + 1):
                # Extract 5x5x5 spatio-temporal block
                spatio_temporal_block = []
                for frame_idx in range(t, t + temporal_window):
                    frame = resized_frames[frame_idx]
                    spatial_block = frame[i:i+block_size, j:j+block_size]
                    spatio_temporal_block.append(spatial_block)
                
                # Convert to numpy array: (5, 5, 5) spatio-temporal volume
                st_volume = np.array(spatio_temporal_block)
                
                # Normalize the volume
                st_volume = st_volume.astype(np.float32) / 255.0
                
                # Apply 3D DCT
                dct_3d = dct(dct(dct(st_volume, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
                
                # Extract low-frequency coefficients
                coeffs = extract_3d_zigzag_coefficients(dct_3d, num_coefficients)
                
                # Compute temporal activity
                block_activity = compute_temporal_activity(st_volume)
                
                all_features.append(coeffs)
                all_activities.append(block_activity)
    
    features_array = np.array(all_features)
    activities_array = np.array(all_activities)
    
    # Filter low-activity blocks if threshold is specified
    if min_activity is not None:
        valid_mask = activities_array >= min_activity
    else:
        valid_mask = np.ones(len(features_array), dtype=bool)
    
    if return_activities:
        return features_array, valid_mask, activities_array
    else:
        return features_array, valid_mask, None


def quantize_features(features: np.ndarray, num_bins: int = 32, 
                      feature_min: Optional[np.ndarray] = None,
                      feature_max: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize features into discrete bins for Naive Bayes classifier.
    
    Uses global min/max from training data if provided, otherwise computes from current features.
    This ensures consistent quantization between training and evaluation.
    
    Args:
        features: Continuous feature values
        num_bins: Number of quantization bins
        feature_min: Optional global minimum per feature (from training)
        feature_max: Optional global maximum per feature (from training)
    
    Returns:
        Tuple of (quantized_features, feature_min, feature_max)
        feature_min and feature_max are computed if not provided, otherwise returned as-is
    """
    # Use provided min/max or compute from current features
    if feature_min is None:
        feature_min = features.min(axis=0, keepdims=True)
    else:
        feature_min = np.array(feature_min).reshape(1, -1)
    
    if feature_max is None:
        feature_max = features.max(axis=0, keepdims=True)
    else:
        feature_max = np.array(feature_max).reshape(1, -1)
    
    # Avoid division by zero
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0
    
    # Normalize features using global min/max
    normalized = (features - feature_min) / feature_range
    
    # Clip to [0, 1] in case test features exceed training range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Quantize to bins
    quantized = (normalized * (num_bins - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, num_bins - 1)
    
    return quantized, feature_min, feature_max


def compute_mutual_information(features: np.ndarray, labels: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """
    Compute mutual information between each feature and class labels.
    
    Implements Equation 1 from Keren (2003):
    MI(f_i, C_j) = P(f_i|C_j) * log(P(f_i|C_j) / P(f_i))
    
    Args:
        features: Feature matrix (N, num_features) - should be quantized/discrete
        labels: Class labels (N,)
        num_bins: Number of bins for feature values (should match quantization)
    
    Returns:
        Mutual information matrix (num_classes, num_features) - MI for each feature-class pair
    """
    N = len(features)
    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    
    # Initialize MI matrix
    mi_matrix = np.zeros((num_classes, num_features))
    
    # Compute P(f_i) - probability of feature i having each value
    feature_probs = np.zeros((num_features, num_bins))
    for f in range(num_features):
        for b in range(num_bins):
            count = np.sum(features[:, f] == b)
            feature_probs[f, b] = count / N if N > 0 else 0
    
    # Compute P(f_i|C_j) and MI for each class
    for c in range(num_classes):
        class_mask = (labels == c)
        class_samples = features[class_mask]
        n_class = len(class_samples)
        
        if n_class == 0:
            continue
        
        # Compute P(f_i|C_j) - probability of feature i = bin b given class j
        feature_given_class = np.zeros((num_features, num_bins))
        for f in range(num_features):
            for b in range(num_bins):
                count = np.sum(class_samples[:, f] == b)
                feature_given_class[f, b] = count / n_class if n_class > 0 else 0
        
        # Compute MI(f_i, C_j) = sum over all bins: P(f_i=b|C_j) * log(P(f_i=b|C_j) / P(f_i=b))
        for f in range(num_features):
            mi = 0.0
            for b in range(num_bins):
                p_f_given_c = feature_given_class[f, b]
                p_f = feature_probs[f, b]
                
                if p_f_given_c > 0 and p_f > 0:
                    mi += p_f_given_c * np.log(p_f_given_c / p_f)
            
            mi_matrix[c, f] = mi
    
    return mi_matrix


def select_features_by_mi(features: np.ndarray, labels: np.ndarray, 
                          num_bins: int = 32, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top features based on mutual information with class labels.
    
    Args:
        features: Feature matrix (N, num_features)
        labels: Class labels (N,)
        num_bins: Number of bins for feature quantization
        top_k: Number of top features to select (if None, selects all with MI > 0)
    
    Returns:
        Tuple of (selected_features, selected_feature_indices)
    """
    mi_matrix = compute_mutual_information(features, labels, num_bins)
    
    # Average MI across classes for each feature
    avg_mi = np.mean(mi_matrix, axis=0)
    
    # Select top features
    if top_k is None:
        # Select all features with positive MI
        selected_indices = np.where(avg_mi > 0)[0]
    else:
        # Select top k features
        selected_indices = np.argsort(avg_mi)[-top_k:][::-1]
    
    selected_features = features[:, selected_indices]
    
    return selected_features, selected_indices


def binarize_features_by_threshold(features: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Binarize features by thresholding coefficient magnitudes.
    
    Args:
        features: Feature matrix (N, num_features) - continuous values
        thresholds: Threshold values for each feature (num_features,)
    
    Returns:
        Binary feature matrix (N, num_features) - 1 if feature > threshold, 0 otherwise
    """
    binary_features = (features > thresholds).astype(np.int32)
    return binary_features


def find_optimal_thresholds_by_mi(features: np.ndarray, labels: np.ndarray, 
                                  num_candidates: int = 100) -> np.ndarray:
    """
    Find optimal thresholds for binarization that maximize mutual information.
    
    For each feature, tests multiple threshold candidates and selects the one
    that maximizes MI with class labels.
    
    Args:
        features: Feature matrix (N, num_features) - continuous values
        labels: Class labels (N,)
        num_candidates: Number of threshold candidates to test per feature
    
    Returns:
        Optimal thresholds for each feature (num_features,)
    """
    num_features = features.shape[1]
    optimal_thresholds = np.zeros(num_features)
    
    for f in range(num_features):
        feature_values = features[:, f]
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        
        # Generate candidate thresholds
        if min_val == max_val:
            optimal_thresholds[f] = min_val
            continue
        
        candidates = np.linspace(min_val, max_val, num_candidates)
        best_mi = -np.inf
        best_threshold = min_val
        
        for threshold in candidates:
            # Binarize with this threshold
            binary = (feature_values > threshold).astype(np.int32)
            
            # Compute MI for this binary feature
            # Treat binary as quantized with 2 bins
            binary_features = binary.reshape(-1, 1)
            mi_matrix = compute_mutual_information(binary_features, labels, num_bins=2)
            avg_mi = np.mean(mi_matrix)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_threshold = threshold
        
        optimal_thresholds[f] = best_threshold
    
    return optimal_thresholds

