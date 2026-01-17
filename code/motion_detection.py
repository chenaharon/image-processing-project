"""
Motion Detection and Classification Module
Detects and classifies different types of motion in video sequences:
- Translation
- Rotation
- Zoom (scale)
- Combined motions
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from scipy.fftpack import dct


class MotionClassifier:
    """
    Classifies motion types in video sequences using optical flow and DCT features.
    """
    
    def __init__(self):
        self.motion_types = {
            0: 'static',
            1: 'translation',
            2: 'rotation',
            3: 'zoom',
            4: 'combined'
        }
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: Previous frame
            frame2: Current frame
        
        Returns:
            Optical flow vectors (H, W, 2)
        """
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Use Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None, 
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow
    
    def classify_motion_type(self, flow: np.ndarray, block_size: int = 32) -> np.ndarray:
        """
        Classify motion type for each block in the frame.
        
        Args:
            flow: Optical flow vectors (H, W, 2)
            block_size: Size of blocks for classification
        
        Returns:
            Motion type labels for each block (H//block_size, W//block_size)
        """
        h, w = flow.shape[:2]
        motion_map = np.zeros((h // block_size, w // block_size), dtype=np.int32)
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block_flow = flow[i:i+block_size, j:j+block_size]
                
                # Extract motion characteristics
                motion_type = self._analyze_block_motion(block_flow)
                motion_map[i // block_size, j // block_size] = motion_type
        
        return motion_map
    
    def _analyze_block_motion(self, block_flow: np.ndarray) -> int:
        """
        Analyze motion in a block and classify its type.
        
        Args:
            block_flow: Optical flow vectors for a block (H, W, 2)
        
        Returns:
            Motion type label (0-4)
        """
        # Extract flow components
        u = block_flow[:, :, 0]  # x-component
        v = block_flow[:, :, 1]  # y-component
        
        # Compute motion magnitude
        magnitude = np.sqrt(u**2 + v**2)
        mean_magnitude = np.mean(magnitude)
        
        # Static if motion is very small
        if mean_magnitude < 0.5:
            return 0  # static
        
        # Compute center of block
        h, w = block_flow.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Compute relative positions
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        rel_x = x_coords - center_x
        rel_y = y_coords - center_y
        
        # Analyze motion patterns
        # Translation: uniform flow direction
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        u_std = np.std(u)
        v_std = np.std(v)
        
        # Rotation: flow perpendicular to radius, magnitude increases with distance
        # Cross product of position and flow
        cross_product = rel_x * v - rel_y * u
        rotation_strength = np.abs(np.mean(cross_product))
        
        # Zoom: flow radial from/to center
        dot_product = rel_x * u + rel_y * v
        zoom_strength = np.abs(np.mean(dot_product))
        
        # Classification logic
        if u_std < 2.0 and v_std < 2.0:
            # Low variance - likely translation
            return 1  # translation
        elif rotation_strength > zoom_strength and rotation_strength > 5.0:
            # Strong rotation pattern
            return 2  # rotation
        elif zoom_strength > rotation_strength and zoom_strength > 5.0:
            # Strong zoom pattern
            return 3  # zoom
        else:
            # Mixed or complex motion
            return 4  # combined
    
    def create_motion_colormap(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Create color map for different motion types.
        
        Returns:
            Dictionary mapping motion type to BGR color
        """
        return {
            0: (0, 0, 0),        # Black - static
            1: (255, 0, 0),      # Blue - translation
            2: (0, 255, 0),      # Green - rotation
            3: (0, 0, 255),      # Red - zoom
            4: (255, 255, 0)     # Cyan - combined
        }
    
    def visualize_motion(self, frame: np.ndarray, motion_map: np.ndarray, 
                        block_size: int = 32) -> np.ndarray:
        """
        Visualize motion types on frame using colors.
        
        Args:
            frame: Original frame
            motion_map: Motion type labels (H//block_size, W//block_size)
            block_size: Size of blocks
        
        Returns:
            Frame with motion regions colored
        """
        colored_frame = frame.copy()
        colormap = self.create_motion_colormap()
        
        h, w = frame.shape[:2]
        map_h, map_w = motion_map.shape
        
        for i in range(map_h):
            for j in range(map_w):
                motion_type = motion_map[i, j]
                color = colormap[motion_type]
                
                # Draw colored rectangle
                y_start = i * block_size
                y_end = min((i + 1) * block_size, h)
                x_start = j * block_size
                x_end = min((j + 1) * block_size, w)
                
                # Blend color with frame
                overlay = colored_frame[y_start:y_end, x_start:x_end].copy()
                colored_rect = np.full((y_end - y_start, x_end - x_start, 3), color, dtype=np.uint8)
                
                # Alpha blending
                alpha = 0.4
                colored_frame[y_start:y_end, x_start:x_end] = (
                    (1 - alpha) * overlay + alpha * colored_rect
                ).astype(np.uint8)
        
        return colored_frame



