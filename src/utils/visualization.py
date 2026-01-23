"""
Visualization Utilities
Creates colored block overlays for video visualization.
Implements Keren 2003 methodology: activity filtering (var>=20), confidence filtering (r>=2.0).
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

# Colors (BGR for OpenCV, matching Keren 2003 Figures 6-7)
COLORS = {
    'hand_wave_side': (0, 255, 255),      # YELLOW (RGB: 255, 255, 0)
    'hand_wave_hello': (255, 0, 0),       # BLUE (RGB: 0, 0, 255)
    'walking': (128, 0, 128),             # PURPLE (RGB: 128, 0, 128)
    'unclassified': (128, 128, 128)       # GRAY (RGB: 128, 128, 128)
}

ACTIVITY_THRESHOLD = 20.0  # Paper: variance >= 20.0
CONFIDENCE_THRESHOLD = 2.0  # Paper: max_prob / min_prob >= 2.0


def create_colored_frame(frame: np.ndarray,
                        block_predictions: List[Dict],
                        upscale: int = 8,
                        id_to_label: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Create colored frame with block overlays.
    
    Args:
        frame: Original frame (64x64 grayscale)
        block_predictions: List of dicts with keys: 'position' (i,j,t), 'prediction' (class_id or -1), 'confidence_ratio'
        upscale: Upscale factor (default: 8, so 64x64 -> 512x512)
        id_to_label: Optional mapping from class_id to label name
    
    Returns:
        Tuple of (colored_frame, stats_dict)
    """
    h, w = frame.shape
    upscaled_h, upscaled_w = h * upscale, w * upscale
    
    # Upscale frame
    frame_upscaled = cv2.resize(frame, (upscaled_w, upscaled_h), interpolation=cv2.INTER_NEAREST)
    colored_frame = cv2.cvtColor(frame_upscaled, cv2.COLOR_GRAY2BGR)
    
    # Create overlay
    overlay = colored_frame.copy()
    
    # Statistics
    stats = {
        'total_blocks': len(block_predictions),
        'active_blocks': 0,
        'classified_blocks': 0,
        'yellow_blocks': 0,
        'blue_blocks': 0,
        'purple_blocks': 0,
        'gray_blocks': 0
    }
    
    # Draw blocks
    for block_info in block_predictions:
        i, j, t = block_info['position']
        prediction = block_info['prediction']
        confidence_ratio = block_info.get('confidence_ratio', 0.0)
        variance = block_info.get('variance', 0.0)
        
        # Activity filtering: skip if variance < 20.0
        if variance < ACTIVITY_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        # Confidence filtering: skip if confidence_ratio < 2.0
        elif confidence_ratio < CONFIDENCE_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        else:
            stats['active_blocks'] += 1
            stats['classified_blocks'] += 1
            
            # Map prediction to color
            if id_to_label is not None and prediction in id_to_label:
                label = id_to_label[prediction]
                if label == 'hand_wave_side':
                    color = COLORS['hand_wave_side']
                    stats['yellow_blocks'] += 1
                elif label == 'hand_wave_hello':
                    color = COLORS['hand_wave_hello']
                    stats['blue_blocks'] += 1
                elif label == 'walking':
                    color = COLORS['walking']
                    stats['purple_blocks'] += 1
                else:
                    color = COLORS['unclassified']
                    stats['gray_blocks'] += 1
            else:
                color = COLORS['unclassified']
                stats['gray_blocks'] += 1
        
        # Draw block (upscaled coordinates)
        x1 = j * upscale
        y1 = i * upscale
        x2 = (j + 5) * upscale
        y2 = (i + 5) * upscale
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend overlay with original (semi-transparent)
    cv2.addWeighted(overlay, 0.6, colored_frame, 0.4, 0, colored_frame)
    
    return colored_frame, stats
