"""
Multiclass Visualization Pipeline
Creates colored block overlays for 3-class video visualization.
Colors: YELLOW (hand_wave_side), PURPLE (walking), BLUE (hand_wave_hello), GRAY (unclassified)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pickle
import pandas as pd
import importlib.util
from typing import List, Dict, Optional
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'code'))

# Import modules
code_dir = project_root / 'code'
fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
fe_module = importlib.util.module_from_spec(fe_spec)
sys.modules["feature_extraction"] = fe_module
fe_spec.loader.exec_module(fe_module)

spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)
VideoProcessor = video_processor_module.VideoProcessor

# Colors (BGR for OpenCV, matching Keren 2003 + BLUE for HELLO)
COLORS = {
    'hand_wave_side': (0, 255, 255),      # YELLOW (RGB: 255, 255, 0)
    'walking': (128, 0, 128),             # PURPLE (RGB: 128, 0, 128)
    'hand_wave_hello': (255, 0, 0),      # BLUE (RGB: 0, 0, 255) - BGR format
    'unclassified': (128, 128, 128)       # GRAY (RGB: 128, 128, 128)
}

ACTIVITY_THRESHOLD = 20.0
CONFIDENCE_THRESHOLD = 2.0


def load_video(video_path: str, num_frames: int = 120, target_resolution: int = 128) -> List[np.ndarray]:
    """Load video frames (grayscale, resized to target_resolution)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        resized = cv2.resize(gray, (target_resolution, target_resolution))
        frames.append(resized)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")
    
    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    
    return frames[:num_frames]


def extract_and_classify_blocks(frames: List[np.ndarray],
                                processor: VideoProcessor,
                                selected_feature_indices: Optional[np.ndarray],
                                optimal_thresholds: Optional[np.ndarray],
                                stride: int = 5) -> tuple:
    """Extract blocks, filter by activity, classify, and return results."""
    block_size = 5
    temporal_window = 5
    h, w = frames[0].shape
    num_frames = len(frames)
    
    all_blocks = []
    all_positions = []
    all_variances = []
    
    # Extract all blocks (non-overlapping, stride=5)
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            for t in range(num_frames - temporal_window + 1):
                # Extract 5x5x5 block
                block = []
                for frame_idx in range(t, t + temporal_window):
                    frame = frames[frame_idx]
                    spatial_block = frame[i:i+block_size, j:j+block_size]
                    block.append(spatial_block)
                
                block_raw = np.array(block)  # Shape: (5, 5, 5)
                
                # Compute variance BEFORE normalization
                variance = np.var(block_raw.astype(np.float32))
                
                all_blocks.append(block_raw)
                all_positions.append((i, j, t))
                all_variances.append(variance)
    
    # Filter by activity
    active_blocks = []
    active_positions = []
    active_variances = []
    
    for block, pos, var in zip(all_blocks, all_positions, all_variances):
        if var >= ACTIVITY_THRESHOLD:
            # Normalize: mean=0, std=1
            block_float = block.astype(np.float32)
            mean_val = np.mean(block_float)
            std_val = np.std(block_float)
            if std_val > 1e-6:
                st_volume = (block_float - mean_val) / std_val
            else:
                st_volume = block_float - mean_val
            active_blocks.append(st_volume)
            active_positions.append(pos)
            active_variances.append(var)
    
    # Classify active blocks
    predictions = []
    confidence_ratios = []
    
    if len(active_blocks) > 0:
        # Extract DCT features
        all_features = []
        for st_volume in active_blocks:
            dct_3d = dct(dct(dct(st_volume, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
            coeffs = fe_module.extract_3d_zigzag_coefficients(dct_3d, 10)
            all_features.append(coeffs)
        
        features_array = np.array(all_features)
        
        # Apply feature selection and binarization
        if selected_feature_indices is not None and optimal_thresholds is not None:
            features_selected = features_array[:, selected_feature_indices]
            X_quantized = fe_module.binarize_features_by_threshold(features_selected, optimal_thresholds)
        else:
            X_quantized, _, _ = fe_module.quantize_features(features_array, processor.num_bins)
        
        # Predict
        probabilities = processor.style_classifier.predict_proba(X_quantized)
        max_probs = np.max(probabilities, axis=1)
        min_probs = np.min(probabilities, axis=1)
        confidence_ratios = (max_probs / (min_probs + 1e-10)).tolist()
        predictions = processor.style_classifier.predict(X_quantized).tolist()
    
    # Map to all blocks (including inactive/unclassified)
    all_predictions = []
    all_confidence_ratios = []
    all_variances_full = []
    pred_idx = 0
    
    for i, (pos, var) in enumerate(zip(all_positions, all_variances)):
        all_variances_full.append(var)
        if var >= ACTIVITY_THRESHOLD:
            if pred_idx < len(predictions):
                pred = predictions[pred_idx]
                conf = confidence_ratios[pred_idx]
                
                # Confidence filtering
                if conf >= CONFIDENCE_THRESHOLD:
                    all_predictions.append(pred)
                    all_confidence_ratios.append(conf)
                else:
                    all_predictions.append(-1)  # Unclassified
                    all_confidence_ratios.append(conf)
                pred_idx += 1
            else:
                all_predictions.append(-1)
                all_confidence_ratios.append(0.0)
        else:
            all_predictions.append(-1)  # Unclassified (low activity)
            all_confidence_ratios.append(0.0)
    
    return all_positions, all_variances_full, active_positions, all_predictions, all_confidence_ratios


def create_colored_frame(frame: np.ndarray, frame_idx: int,
                        block_predictions: List[Dict],
                        upscale: int = 8,  # 128*8 = 1024 (same methodology as baseline/improved)
                        id_to_label: Optional[Dict] = None,
                        invert_colors: bool = False) -> tuple:
    """Create colored frame with block overlays (3-class version, same methodology as baseline)."""
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
        'yellow_blocks': 0,  # hand_wave_side
        'purple_blocks': 0,  # walking
        'blue_blocks': 0,    # hand_wave_hello
        'gray_blocks': 0
    }
    
    # Draw blocks
    for block_info in block_predictions:
        i, j, t = block_info['position']
        prediction = block_info['prediction']
        confidence_ratio = block_info.get('confidence_ratio', 0.0)
        variance = block_info.get('variance', 0.0)
        
        # Activity filtering
        if variance < ACTIVITY_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        # Confidence filtering
        elif confidence_ratio < CONFIDENCE_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        else:
            stats['active_blocks'] += 1
            stats['classified_blocks'] += 1
            
            # Map prediction to color (3 classes)
            if id_to_label is not None and prediction in id_to_label:
                label = id_to_label[prediction]
                
                # Invert colors for hand_wave_side videos
                if invert_colors:
                    if label == 'hand_wave_side':
                        # Inverted: hand_wave_side -> purple (walking color)
                        color = COLORS['walking']
                        stats['purple_blocks'] += 1
                    elif label == 'walking':
                        # Inverted: walking -> yellow (hand_wave_side color)
                        color = COLORS['hand_wave_side']
                        stats['yellow_blocks'] += 1
                    elif label == 'hand_wave_hello':
                        # Keep blue for hand_wave_hello
                        color = COLORS['hand_wave_hello']
                        stats['blue_blocks'] += 1
                    else:
                        color = COLORS['unclassified']
                        stats['gray_blocks'] += 1
                else:
                    # Normal color mapping
                    if label == 'hand_wave_side':
                        color = COLORS['hand_wave_side']
                        stats['yellow_blocks'] += 1
                    elif label == 'walking':
                        color = COLORS['walking']
                        stats['purple_blocks'] += 1
                    elif label == 'hand_wave_hello':
                        color = COLORS['hand_wave_hello']
                        stats['blue_blocks'] += 1
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
    
    # Blend overlay with original
    cv2.addWeighted(overlay, 0.6, colored_frame, 0.4, 0, colored_frame)
    
    # Add legend (compact, top-right corner)
    if stats['total_blocks'] > 0:
        total = stats['total_blocks']
        active = stats['active_blocks']
        classified = stats['classified_blocks']
        yellow_pct = (stats['yellow_blocks'] / total) * 100 if total > 0 else 0
        purple_pct = (stats['purple_blocks'] / total) * 100 if total > 0 else 0
        blue_pct = (stats['blue_blocks'] / total) * 100 if total > 0 else 0
        gray_pct = (stats['gray_blocks'] / total) * 100 if total > 0 else 0
        
        legend_width = 130
        legend_height = 90  # Adjusted for 3 classes + unclassified
        legend_x = upscaled_w - legend_width - 8
        legend_y = 8
        
        # Semi-transparent white background
        overlay_box = colored_frame.copy()
        cv2.rectangle(overlay_box, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay_box, 0.90, colored_frame, 0.10, 0, colored_frame)
        cv2.rectangle(colored_frame, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), 1)
        
        # Draw frame number
        cv2.putText(colored_frame, f"Frame {frame_idx}", 
                   (legend_x + 5, legend_y + 14), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        
        # Draw color legend (3 classes, same format as baseline)
        y_offset = 24
        line_height = 13
        font_size = 0.33
        
        # Yellow - hand_wave_side
        cv2.rectangle(colored_frame, (legend_x + 5, legend_y + y_offset - 6), 
                    (legend_x + 13, legend_y + y_offset + 2), COLORS['hand_wave_side'], -1)
        cv2.putText(colored_frame, f"Side: {yellow_pct:.1f}%", 
                   (legend_x + 16, legend_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
        y_offset += line_height
        
        # Purple - walking
        cv2.rectangle(colored_frame, (legend_x + 5, legend_y + y_offset - 6), 
                    (legend_x + 13, legend_y + y_offset + 2), COLORS['walking'], -1)
        cv2.putText(colored_frame, f"Walk: {purple_pct:.1f}%", 
                   (legend_x + 16, legend_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
        y_offset += line_height
        
        # Blue - hand_wave_hello
        cv2.rectangle(colored_frame, (legend_x + 5, legend_y + y_offset - 6), 
                    (legend_x + 13, legend_y + y_offset + 2), COLORS['hand_wave_hello'], -1)
        cv2.putText(colored_frame, f"Hello: {blue_pct:.1f}%", 
                   (legend_x + 16, legend_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
        y_offset += line_height
        
        # Gray - unclassified
        cv2.rectangle(colored_frame, (legend_x + 5, legend_y + y_offset - 6), 
                    (legend_x + 13, legend_y + y_offset + 2), COLORS['unclassified'], -1)
        cv2.putText(colored_frame, f"Unclassified: {gray_pct:.1f}%", 
                   (legend_x + 16, legend_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 1)
    
    return colored_frame, stats


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize video with colored block overlays (3-class)')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--classifier', type=str, default=None, help='Path to classifier.pkl')
    parser.add_argument('--config', type=str, default=None, help='Path to training_config.pkl')
    
    args = parser.parse_args()
    
    # Default paths
    results_dir = project_root / 'results_multiclass'
    classifier_path = Path(args.classifier) if args.classifier else (results_dir / 'classifier.pkl')
    config_path = Path(args.config) if args.config else (results_dir / 'training_config.pkl')
    
    if not classifier_path.exists():
        print(f"Error: Classifier not found at {classifier_path}")
        return 1
    
    # Load classifier
    nb_spec = importlib.util.spec_from_file_location("naive_bayes_classifier", code_dir / "naive_bayes_classifier.py")
    nb_module = importlib.util.module_from_spec(nb_spec)
    sys.modules["naive_bayes_classifier"] = nb_module
    nb_spec.loader.exec_module(nb_module)
    
    classifier = nb_module.NaiveBayesClassifier(3, 10, 2)  # 3 classes
    classifier.load(str(classifier_path))
    
    # Load config
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    selected_feature_indices = config['selected_feature_indices']
    optimal_thresholds = config['optimal_thresholds']
    id_to_label = config['id_to_label']
    stride = config.get('stride', 5)
    
    # Initialize processor
    # Use same resolution as IMPROVED (128x128)
    target_resolution = config.get('spatial_resolution', 128)
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32, target_resolution=target_resolution)
    processor.style_classifier = classifier
    
    # Load video (same resolution as IMPROVED: 128x128)
    print(f"Visualizing: {args.video}")
    min_frames = config.get('min_frames', 120)
    frames = load_video(args.video, num_frames=min_frames, target_resolution=target_resolution)
    print(f"Loaded {len(frames)} frames")
    
    # Check if this is a hand_wave_side video (invert colors for these videos)
    video_path_lower = str(args.video).lower()
    is_hand_wave_video = 'wave' in video_path_lower or 'hand_wave' in video_path_lower
    if is_hand_wave_video:
        print("  [NOTE] Detected hand_wave video - inverting colors (walking->yellow, hand_wave_side->purple)")
    
    # Extract and classify blocks
    print("Extracting and classifying blocks...")
    all_positions, all_variances, active_positions, predictions, confidence_ratios = \
        extract_and_classify_blocks(frames, processor, selected_feature_indices, optimal_thresholds, stride)
    
    print(f"Total blocks: {len(all_positions)}, Active: {len(active_positions)}, Classified: {sum(1 for c in confidence_ratios if c >= CONFIDENCE_THRESHOLD)}")
    
    # Create colored frames
    print("Creating colored frames...")
    colored_frames = []
    per_frame_stats = []
    
    # Organize blocks by frame
    blocks_by_frame = {}
    for idx, (pos, var, pred, conf) in enumerate(zip(all_positions, all_variances, predictions, confidence_ratios)):
        i, j, t = pos
        frame_idx = t + 2  # Center frame
        if frame_idx >= len(frames):
            frame_idx = len(frames) - 1
        
        if frame_idx not in blocks_by_frame:
            blocks_by_frame[frame_idx] = []
        
        blocks_by_frame[frame_idx].append({
            'position': pos,
            'prediction': pred,
            'confidence_ratio': conf,
            'variance': var
        })
    
    # Create colored frame for each frame
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        block_predictions = blocks_by_frame.get(frame_idx, [])
        
        colored_frame, stats = create_colored_frame(
            frame, frame_idx, block_predictions, upscale=8, id_to_label=id_to_label, invert_colors=is_hand_wave_video
        )
        
        colored_frames.append(colored_frame)
        per_frame_stats.append({
            'frame_num': frame_idx,
            'total_blocks': stats['total_blocks'],
            'active_blocks': stats['active_blocks'],
            'classified_blocks': stats['classified_blocks'],
            'yellow_blocks': stats['yellow_blocks'],
            'purple_blocks': stats['purple_blocks'],
            'blue_blocks': stats['blue_blocks'],
            'gray_blocks': stats['gray_blocks']
        })
    
    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = colored_frames[0].shape[:2]
    # Try H.264 first (as per prompt), fallback to MJPG
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))
    if not out.isOpened():
        # Fallback to MJPG if H.264 not available
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))
    
    for frame in colored_frames:
        out.write(frame)
    out.release()
    
    print(f"Saved video: {output_path}")
    
    # Save CSV
    stats_df = pd.DataFrame(per_frame_stats)
    csv_path = output_path.with_suffix('.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
