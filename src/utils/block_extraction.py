"""
Block Extraction Utilities
Extracts 5x5x5 spatio-temporal blocks from videos.
Based on Keren (2003) paper methodology.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.fftpack import dct


def extract_blocks_non_overlapping(video_frames: List[np.ndarray],
                                   block_size: int = 5,
                                   temporal_window: int = 5) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract non-overlapping 5x5x5 spatio-temporal blocks.
    
    Args:
        video_frames: List of video frames (grayscale, 64x64)
        block_size: Spatial block size (default: 5)
        temporal_window: Temporal window size (default: 5)
    
    Returns:
        Tuple of (blocks, positions) where positions are (i, j, t) tuples
    """
    if len(video_frames) == 0:
        return [], []
    
    h, w = video_frames[0].shape
    blocks = []
    positions = []
    
    # Non-overlapping: stride = block_size
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            for t in range(len(video_frames) - temporal_window + 1):
                # Extract 5x5x5 spatio-temporal block
                block = []
                for frame_idx in range(t, t + temporal_window):
                    frame = video_frames[frame_idx]
                    spatial_block = frame[i:i+block_size, j:j+block_size]
                    block.append(spatial_block)
                
                st_volume = np.array(block)  # Shape: (5, 5, 5)
                blocks.append(st_volume)
                positions.append((i, j, t))
    
    return blocks, positions


def extract_blocks_overlapping(video_frames: List[np.ndarray],
                              block_size: int = 5,
                              stride: int = 2,
                              temporal_window: int = 5) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract overlapping 5x5x5 spatio-temporal blocks with specified stride.
    
    Args:
        video_frames: List of video frames (grayscale, 64x64)
        block_size: Spatial block size (default: 5)
        stride: Spatial stride (default: 2 for overlapping)
        temporal_window: Temporal window size (default: 5)
    
    Returns:
        Tuple of (blocks, positions) where positions are (i, j, t) tuples
    """
    if len(video_frames) == 0:
        return [], []
    
    h, w = video_frames[0].shape
    blocks = []
    positions = []
    
    # Overlapping: stride < block_size
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            for t in range(len(video_frames) - temporal_window + 1):
                # Extract 5x5x5 spatio-temporal block
                block = []
                for frame_idx in range(t, t + temporal_window):
                    frame = video_frames[frame_idx]
                    spatial_block = frame[i:i+block_size, j:j+block_size]
                    block.append(spatial_block)
                
                st_volume = np.array(block)  # Shape: (5, 5, 5)
                blocks.append(st_volume)
                positions.append((i, j, t))
    
    return blocks, positions


def normalize_block(block: np.ndarray) -> np.ndarray:
    """
    Normalize block to zero mean and unit variance (Keren 2003 methodology).
    
    Args:
        block: 5x5x5 spatio-temporal block
    
    Returns:
        Normalized block (mean=0, std=1)
    """
    block_float = block.astype(np.float32)
    mean_val = np.mean(block_float)
    std_val = np.std(block_float)
    
    if std_val > 1e-6:
        return (block_float - mean_val) / std_val
    else:
        return block_float - mean_val  # If std=0, just center


def compute_block_variance(block: np.ndarray) -> float:
    """
    Compute variance of block BEFORE normalization.
    Used for activity filtering (variance >= 20.0).
    
    Args:
        block: 5x5x5 spatio-temporal block
    
    Returns:
        Variance value
    """
    return np.var(block.astype(np.float32))
