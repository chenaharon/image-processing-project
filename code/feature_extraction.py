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


def extract_spatial_temporal_features(video_frames: List[np.ndarray], 
                                     block_size: int = 5,
                                     num_coefficients: int = 10,
                                     temporal_window: int = 5) -> np.ndarray:
    """
    Extract spatio-temporal features from video frames using 5x5x5 neighborhoods.
    
    Implements the methodology from Keren (2003): "Recognizing image 'style' and 
    activities in video using local features and naive Bayes".
    
    Args:
        video_frames: List of video frames (should be resized to 64x64)
        block_size: Size of spatial blocks (default: 5x5 as in paper)
        num_coefficients: Number of DCT coefficients per block
        temporal_window: Number of frames in temporal window (default: 5 as in paper)
    
    Returns:
        Array of features (N, num_coefficients) - one feature vector per 5x5x5 neighborhood
    """
    if len(video_frames) < temporal_window:
        raise ValueError(f"Need at least {temporal_window} frames for spatio-temporal features")
    
    # Resize frames to 64x64 as in the paper
    resized_frames = []
    for frame in video_frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        resized = cv2.resize(gray, (64, 64))
        resized_frames.append(resized)
    
    all_features = []
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
                
                # Normalize the volume
                st_volume = st_volume.astype(np.float32) / 255.0
                
                # Paper methodology: Extract DCT from center frame's 5x5 spatial block
                # The 5x5x5 neighborhood provides temporal context for classification
                center_frame_idx = temporal_window // 2
                center_block = st_volume[center_frame_idx]  # 5x5 block
                
                # Apply 2D DCT to the center block
                dct_block = dct(dct(center_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                
                # Extract low-frequency coefficients (zigzag pattern)
                # Paper uses first num_coefficients (typically 10)
                coeffs = extract_zigzag_coefficients(dct_block, num_coefficients)
                
                all_features.append(coeffs)
    
    return np.array(all_features)


def quantize_features(features: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """
    Quantize features into discrete bins for Naive Bayes classifier.
    
    Args:
        features: Continuous feature values
        num_bins: Number of quantization bins
    
    Returns:
        Quantized features
    """
    # Normalize features to [0, 1]
    feature_min = features.min(axis=0, keepdims=True)
    feature_max = features.max(axis=0, keepdims=True)
    
    # Avoid division by zero
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0
    
    normalized = (features - feature_min) / feature_range
    
    # Quantize to bins
    quantized = (normalized * (num_bins - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, num_bins - 1)
    
    return quantized

