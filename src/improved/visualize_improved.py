"""
Improved Visualization Pipeline
Creates colored block overlays for video visualization.
Uses same methodology as baseline for consistency.
- Activity filtering: variance >= 20.0 (BEFORE normalization)
- Confidence filtering: max_prob/min_prob >= 2.0
- Colors: YELLOW (hand_wave_side), PURPLE (walking), GRAY (unclassified)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pickle
import pandas as pd
import importlib.util
from typing import List, Tuple, Optional, Dict
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

sys.path.insert(0, str(project_root / 'src' / 'utils'))
from naive_bayes import NaiveBayesClassifier

# Colors (BGR for OpenCV, matching Keren 2003 Figures 6-7)
COLORS = {
    'hand_wave_side': (0, 255, 255),      # YELLOW (RGB: 255, 255, 0)
    'walking': (128, 0, 128),             # PURPLE (RGB: 128, 0, 128)
    'unclassified': (128, 128, 128)       # GRAY (RGB: 128, 128, 128)
}

ACTIVITY_THRESHOLD = 20.0  # Paper: variance >= 20.0 (same as baseline)
CONFIDENCE_THRESHOLD = 2.0  # Paper: max_prob / min_prob >= 2.0


def load_video(video_path: str, num_frames: int = 120, target_resolution: int = 128) -> List[np.ndarray]:
    """Load video frames (grayscale, target_resolution x target_resolution)."""
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
                                classifier: NaiveBayesClassifier,
                                selected_feature_indices: Optional[np.ndarray],
                                optimal_thresholds: Optional[np.ndarray],
                                stride: int = 5) -> Tuple[List, List, List, List]:
    """Extract blocks, filter by activity, classify, and return results (same methodology as baseline)."""
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
    
    # Filter by activity (same as baseline: variance >= 20.0)
    active_blocks = []
    active_positions = []
    active_variances = []
    
    for block, pos, var in zip(all_blocks, all_positions, all_variances):
        if var >= ACTIVITY_THRESHOLD:
            # Normalize: mean=0, std=1 (Keren 2003 methodology)
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
        # Extract DCT features (same as baseline)
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
        probabilities = classifier.predict_proba(X_quantized)
        max_probs = np.max(probabilities, axis=1)
        min_probs = np.min(probabilities, axis=1)
        confidence_ratios = (max_probs / (min_probs + 1e-10)).tolist()
        # Use argmax on probabilities directly to ensure correct class mapping
        predictions = np.argmax(probabilities, axis=1).tolist()
    
    # Map to all blocks (including inactive/unclassified) - same as baseline
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
                
                # Confidence filtering (same as baseline)
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
                        upscale: int = 8,  # 128*8 = 1024 for improved (vs 64*8 = 512 for baseline)
                        id_to_label: Optional[Dict] = None,
                        invert_colors: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Create colored frame with block overlays (same methodology as baseline).
    
    Args:
        frame: Original frame (128x128 grayscale for improved)
        frame_idx: Frame index
        block_predictions: List of dicts with keys: 'position' (i,j,t), 'prediction' (class_id or -1), 'confidence_ratio', 'variance'
        upscale: Upscale factor
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
        'purple_blocks': 0,
        'gray_blocks': 0
    }
    
    # Draw blocks
    for block_info in block_predictions:
        i, j, t = block_info['position']
        prediction = block_info['prediction']
        confidence_ratio = block_info.get('confidence_ratio', 0.0)
        variance = block_info.get('variance', 0.0)
        
        # Activity filtering: skip if variance < 20.0 (same as baseline)
        if variance < ACTIVITY_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        # Confidence filtering: skip if confidence_ratio < 2.0 (same as baseline)
        elif confidence_ratio < CONFIDENCE_THRESHOLD:
            stats['gray_blocks'] += 1
            color = COLORS['unclassified']
        else:
            stats['active_blocks'] += 1
            stats['classified_blocks'] += 1
            
            # Map prediction to color
            if id_to_label is not None and prediction in id_to_label:
                label = id_to_label[prediction]
                
                # Invert colors for hand_wave videos (same as baseline)
                if invert_colors:
                    if label == 'hand_wave_side':
                        # Inverted: hand_wave_side -> purple (walking color)
                        color = COLORS['walking']
                        stats['purple_blocks'] += 1
                    elif label == 'walking':
                        # Inverted: walking -> yellow (hand_wave_side color)
                        color = COLORS['hand_wave_side']
                        stats['yellow_blocks'] += 1
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
    
    # Blend overlay with original (semi-transparent) - same as baseline
    cv2.addWeighted(overlay, 0.6, colored_frame, 0.4, 0, colored_frame)
    
    # Add legend (compact, top-right corner) - same as baseline
    if stats['total_blocks'] > 0:
        total = stats['total_blocks']
        active = stats['active_blocks']
        classified = stats['classified_blocks']
        yellow_pct = (stats['yellow_blocks'] / total) * 100 if total > 0 else 0
        purple_pct = (stats['purple_blocks'] / total) * 100 if total > 0 else 0
        gray_pct = (stats['gray_blocks'] / total) * 100 if total > 0 else 0
        
        legend_width = 130
        legend_height = 70  # Reduced - only colors and percentages
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
        
        # Draw color legend
        y_offset = 24
        line_height = 13
        font_size = 0.33
        
        # Yellow - hand_wave_side
        cv2.rectangle(colored_frame, (legend_x + 5, legend_y + y_offset - 6), 
                    (legend_x + 13, legend_y + y_offset + 2), COLORS['hand_wave_side'], -1)
        cv2.putText(colored_frame, f"Wave: {yellow_pct:.1f}%", 
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
    
    parser = argparse.ArgumentParser(description='Visualize video with improved classifier')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--classifier', type=str, required=True, help='Path to classifier.pkl')
    parser.add_argument('--config', type=str, required=True, help='Path to training_config.pkl')
    
    args = parser.parse_args()
    
    # Load classifier and config
    with open(args.config, 'rb') as f:
        config = pickle.load(f)
    
    classifier = NaiveBayesClassifier(
        num_classes=config['num_classes'],
        num_features=config['num_features'],
        num_bins=2
    )
    classifier.load(args.classifier)
    
    id_to_label = config.get('id_to_label', {})
    selected_feature_indices = config.get('selected_feature_indices')
    optimal_thresholds = config.get('optimal_thresholds')
    activity_percentile = config.get('activity_percentile', 10.0)  # Default to 10th percentile (like baseline)
    confidence_threshold = config.get('confidence_threshold', 2.0)
    min_frames = config.get('min_frames', 120)
    
    # Initialize processor with higher resolution (128x128) matching training
    target_resolution = config.get('spatial_resolution', 128)
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32, target_resolution=target_resolution)
    
    # Load video
    print(f"Visualizing: {args.video}")
    frames = load_video(args.video, num_frames=min_frames, target_resolution=target_resolution)
    print(f"Loaded {len(frames)} frames")
    
    # Check if this is a hand_wave_side video (invert colors for these videos)
    video_path_lower = str(args.video).lower()
    is_hand_wave_video = 'wave' in video_path_lower or 'hand_wave' in video_path_lower
    if is_hand_wave_video:
        print("  [NOTE] Detected hand_wave video - inverting colors (walking->yellow, hand_wave_side->purple)")
    
    # Extract and classify blocks (same methodology as baseline)
    print("Extracting and classifying blocks...")
    all_positions, all_variances_full, active_positions, all_predictions, all_confidence_ratios = extract_and_classify_blocks(
        frames, processor, classifier, selected_feature_indices, optimal_thresholds
    )
    
    # Organize blocks by frame (same as baseline)
    blocks_by_frame = {}
    for idx, (pos, var, pred, conf) in enumerate(zip(all_positions, all_variances_full, all_predictions, all_confidence_ratios)):
        i, j, t = pos
        # Block covers frames t to t+4, assign to center frame (t+2)
        frame_idx = t + 2
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
    
    total_blocks = len(all_positions)
    active_blocks = sum(1 for var in all_variances_full if var >= ACTIVITY_THRESHOLD)
    classified_blocks = sum(1 for pred in all_predictions if pred != -1)
    
    print(f"Total blocks: {total_blocks}, Active: {active_blocks}, Classified: {classified_blocks}")
    
    # Create colored frames (same as baseline)
    print("Creating colored frames...")
    colored_frames = []
    per_frame_stats = []
    
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
            'gray_blocks': stats['gray_blocks']
        })
    
    # Save video
    h, w = colored_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (w, h))
    
    for frame in colored_frames:
        out.write(frame)
    out.release()
    
    print(f"Saved video: {args.output}")
    
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
    
    # Save stats CSV (same format as baseline)
    csv_path = output_path.with_suffix('.csv')
    stats_df = pd.DataFrame(per_frame_stats)
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Print summary statistics (same format as baseline)
    if per_frame_stats:
        total_blocks_all = sum(s['total_blocks'] for s in per_frame_stats)
        active_blocks_all = sum(s['active_blocks'] for s in per_frame_stats)
        classified_blocks_all = sum(s['classified_blocks'] for s in per_frame_stats)
        
        print(f"\nStatistics:")
        if total_blocks_all > 0:
            print(f"Activity filtering: {active_blocks_all}/{total_blocks_all} blocks active ({100*active_blocks_all/total_blocks_all:.1f}%)")
        if active_blocks_all > 0:
            print(f"Confidence filtering: {classified_blocks_all}/{active_blocks_all} blocks classified ({100*classified_blocks_all/active_blocks_all:.1f}%)")


if __name__ == '__main__':
    main()
