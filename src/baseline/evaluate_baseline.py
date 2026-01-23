"""
Baseline Evaluation Pipeline
Evaluates trained classifier with activity/confidence filtering.
Generates all 6 graphs and CSV/TXT files as per prompt specifications.

Implements Keren (2003) evaluation:
- Activity filtering: variance >= 20.0 (handled in feature extraction)
- Confidence filtering: max_prob/min_prob >= 2.0
- Block/frame/video level accuracy
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.max_open_warning'] = 0

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'code'))

# Import video processor and feature extraction
code_dir = project_root / 'code'
spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)
VideoProcessor = video_processor_module.VideoProcessor

fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
fe_module = importlib.util.module_from_spec(fe_spec)
sys.modules["feature_extraction"] = fe_module
fe_spec.loader.exec_module(fe_module)

nb_spec = importlib.util.spec_from_file_location("naive_bayes_classifier", code_dir / "naive_bayes_classifier.py")
nb_module = importlib.util.module_from_spec(nb_spec)
sys.modules["naive_bayes_classifier"] = nb_module
nb_spec.loader.exec_module(nb_module)

# Import plot generation
plot_spec = importlib.util.spec_from_file_location("generate_plots", code_dir / "generate_plots.py")
plot_module = importlib.util.module_from_spec(plot_spec)
sys.modules["generate_plots"] = plot_module
plot_spec.loader.exec_module(plot_module)

# Import metrics utilities
sys.path.insert(0, str(project_root / 'src' / 'utils'))
from metrics import (
    compute_block_level_accuracy,
    compute_frame_level_accuracy,
    compute_video_level_accuracy
)

# Constants - will be loaded from config
CONFIDENCE_THRESHOLD = 2.0  # Confidence ratio threshold

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Additional imports for evaluation functions
from collections import defaultdict
from scipy.fftpack import dct
import cv2
from typing import List, Dict, Optional, Tuple


def _extract_and_classify_blocks(
    frames: List[np.ndarray],
    processor,
    selected_feature_indices: Optional[np.ndarray],
    optimal_thresholds: Optional[np.ndarray],
    stride: int = 5,
    activity_threshold: float = 20.0
) -> Tuple[List, List, List, List, List]:
    """
    Extract blocks, filter by activity, classify, and return results.
    Same logic as in visualize_baseline.py - processes frames resized to target_resolution.
    """
    # First, resize frames to target_resolution (same as training)
    resized_frames = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        resized = cv2.resize(gray, (processor.target_resolution, processor.target_resolution))
        resized_frames.append(resized)
    
    block_size = 5
    temporal_window = 5
    h, w = resized_frames[0].shape
    num_frames = len(resized_frames)
    
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
                    frame = resized_frames[frame_idx]
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
        if var >= activity_threshold:
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
        predictions = np.argmax(probabilities, axis=1).tolist()
    
    # Map to all blocks (including inactive/unclassified)
    all_predictions = []
    all_confidence_ratios = []
    all_variances_full = all_variances.copy()
    pred_idx = 0
    
    for i, (pos, var) in enumerate(zip(all_positions, all_variances)):
        if var >= activity_threshold:
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


def evaluate_on_set(
    processor,
    video_paths: List[str],
    labels: List[str],
    label_to_id: Dict[str, int],
    set_name: str = "Set",
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None,
    min_frames: int = 120,
    selected_feature_indices: Optional[np.ndarray] = None,
    optimal_thresholds: Optional[np.ndarray] = None,
    confidence_threshold: float = 2.0,
    stride: int = 5,
    activity_threshold: float = 20.0
) -> Dict:
    """
    Evaluate classifier on a set of videos.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating on {set_name} Set")
    print(f"{'=' * 60}")
    
    all_block_predictions = []
    all_block_labels = []
    all_confidence_ratios = []
    
    # Per-frame data for frame-level accuracy
    block_predictions_per_frame = []
    block_labels_per_frame = []
    block_confidence_per_frame = []
    
    video_level_predictions = []
    video_level_labels = []
    video_stats = []
    video_predictions_list = []
    
    print(f"Processing {len(video_paths)} videos...", flush=True)
    
    for idx, (video_path, label_str) in enumerate(zip(video_paths, labels), 1):
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            print(f"[{idx}/{len(video_paths)}] {video_path_obj.name}... SKIP (file not found)")
            continue
        
        label_id = label_to_id[label_str]
        
        try:
            print(f"[{idx}/{len(video_paths)}] {video_path_obj.name}... Loading video...", flush=True)
            # Load video
            frames = processor.load_video(str(video_path), max_frames=min_frames, start_from_center=True)
            
            # Convert to grayscale if needed
            if len(frames) > 0 and len(frames[0].shape) == 3:
                frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]
            
            if len(frames) == 0:
                print("SKIP (no frames)")
                continue
            
            print(f"  Extracting blocks ({len(frames)} frames)...", flush=True)
            # Extract blocks and classify
            all_positions, all_variances, active_positions, all_predictions, all_confidence_ratios = \
                _extract_and_classify_blocks(
                    frames,
                    processor,
                    selected_feature_indices=selected_feature_indices,
                    optimal_thresholds=optimal_thresholds,
                    stride=stride,
                    activity_threshold=activity_threshold
                )
            print(f"  Classified {len([p for p in all_predictions if p >= 0])} blocks...", flush=True)
            
            # Organize blocks by frame for frame-level accuracy
            frames_dict = defaultdict(lambda: {'preds': [], 'labels': [], 'confs': []})
            
            for pos, pred, conf in zip(all_positions, all_predictions, all_confidence_ratios):
                spatial_i, spatial_j, temporal_t = pos
                # Only include classified blocks (pred >= 0)
                if pred >= 0:
                    frames_dict[temporal_t]['preds'].append(pred)
                    frames_dict[temporal_t]['labels'].append(label_id)
                    frames_dict[temporal_t]['confs'].append(conf)
            
            # Add to per-frame lists
            num_frames = len(frames)
            for t in range(num_frames):
                if t in frames_dict:
                    block_predictions_per_frame.append(frames_dict[t]['preds'])
                    block_labels_per_frame.append(frames_dict[t]['labels'])
                    block_confidence_per_frame.append(frames_dict[t]['confs'])
                else:
                    block_predictions_per_frame.append([])
                    block_labels_per_frame.append([])
                    block_confidence_per_frame.append([])
            
            # Collect all block predictions (only classified ones, pred >= 0)
            # Note: all_predictions already filtered by activity AND confidence thresholds in _extract_and_classify_blocks
            classified_predictions = [p for p in all_predictions if p >= 0]
            classified_confidences = [c for p, c in zip(all_predictions, all_confidence_ratios) if p >= 0]
            
            # Only add blocks that passed both activity and confidence thresholds
            all_block_predictions.extend(classified_predictions)
            all_block_labels.extend([label_id] * len(classified_predictions))
            all_confidence_ratios.extend(classified_confidences)
            
            # Video-level prediction (majority vote)
            video_pred, video_conf = compute_video_level_accuracy(
                classified_predictions, label_id, classified_confidences, confidence_threshold
            )
            
            video_level_predictions.append(video_pred)
            video_level_labels.append(label_id)
            
            correct = "OK" if video_pred == label_id else "X"
            id_to_label = {v: k for k, v in label_to_id.items()}
            video_predictions_list.append({
                'video_name': video_path_obj.name,
                'true_label': label_str,
                'pred_label': id_to_label.get(video_pred, "unclassified") if video_pred >= 0 else "unclassified",
                'correct': video_pred == label_id
            })
            
            # Per-video statistics
            total_blocks_video = len(all_positions)
            classified_blocks_video = len(classified_predictions)
            unclassified_blocks_video = total_blocks_video - classified_blocks_video
            
            video_stats.append({
                'video_name': video_path_obj.name,
                'true_label': label_str,
                'total_blocks': total_blocks_video,
                'classified_blocks': classified_blocks_video,
                'unclassified_blocks': unclassified_blocks_video,
                'classified_percentage': 100.0 * classified_blocks_video / total_blocks_video if total_blocks_video > 0 else 0.0
            })
            
            # Calculate per-video accuracy
            correct_blocks = sum(1 for p, l in zip(classified_predictions, [label_id]*len(classified_predictions)) if p == l)
            video_block_acc = correct_blocks / classified_blocks_video if classified_blocks_video > 0 else 0.0
            
            print(f"  {correct} true:{label_str} -> pred:{video_predictions_list[-1]['pred_label']} "
                  f"(conf:{video_conf:.2f}, blocks:{classified_blocks_video}/{total_blocks_video}, "
                  f"block_acc:{video_block_acc:.2%})", flush=True)
            
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    # Compute accuracies
    # Note: all_block_predictions already filtered by activity AND confidence thresholds
    # So we don't need to filter again by confidence in compute_block_level_accuracy
    if len(all_block_predictions) > 0:
        # All blocks in all_block_predictions already passed confidence threshold
        # So we pass None for confidence_ratios to skip filtering
        block_acc, num_correct_blocks, num_classified_blocks = compute_block_level_accuracy(
            all_block_predictions, all_block_labels, confidence_ratios=None, confidence_threshold=confidence_threshold
        )
    else:
        block_acc, num_correct_blocks, num_classified_blocks = 0.0, 0, 0
    
    if len(block_predictions_per_frame) > 0:
        frame_acc, num_correct_frames, num_total_frames = compute_frame_level_accuracy(
            block_predictions_per_frame, block_labels_per_frame, block_confidence_per_frame, confidence_threshold
        )
    else:
        frame_acc, num_correct_frames, num_total_frames = 0.0, 0, 0
    
    video_correct = sum(1 for p, l in zip(video_level_predictions, video_level_labels) if p == l)
    video_acc = video_correct / len(video_level_predictions) if len(video_level_predictions) > 0 else 0.0
    
    total_blocks = len(all_block_predictions)
    classified_blocks = num_classified_blocks
    unclassified_blocks = total_blocks - classified_blocks
    
    # Print summary
    print(f"\n{set_name} Set Results:")
    print(f"  Block-level accuracy: {block_acc:.2%} ({num_correct_blocks}/{num_classified_blocks} blocks)")
    print(f"  Frame-level accuracy: {frame_acc:.2%} ({num_correct_frames}/{num_total_frames} frames)")
    print(f"  Video-level accuracy: {video_acc:.2%} ({video_correct}/{len(video_level_predictions)} videos)")
    print(f"  Total blocks: {total_blocks}, Classified: {classified_blocks}, Unclassified: {unclassified_blocks}")
    
    return {
        'block_predictions': all_block_predictions,
        'block_labels': all_block_labels,
        'confidence_ratios': all_confidence_ratios,
        'video_predictions': video_level_predictions,
        'video_labels': video_level_labels,
        'video_stats': video_stats,
        'video_predictions_list': video_predictions_list,
        'block_accuracy': block_acc,
        'frame_accuracy': frame_acc,
        'video_accuracy': video_acc,
        'total_blocks': total_blocks,
        'classified_blocks': classified_blocks,
        'unclassified_blocks': unclassified_blocks,
        'num_correct_blocks': num_correct_blocks,
        'num_classified_blocks': num_classified_blocks,
        'num_correct_frames': num_correct_frames,
        'num_total_frames': num_total_frames,
        'video_correct': video_correct
    }


def main():
    """Main evaluation function."""
    import os
    
    print("=" * 60)
    print("BASELINE PIPELINE - Evaluation")
    print("=" * 60)
    print("\nStarting evaluation process...")
    print("Methodology: Exact reproduction of Keren 2003 paper")
    
    # Check if we're evaluating on unseen data
    unseen_data_path = os.environ.get('UNSEEN_DATA', None)
    is_unseen_evaluation = unseen_data_path is not None and Path(unseen_data_path).exists()
    
    if is_unseen_evaluation:
        print("\n[MODE] Evaluating on UNSEEN DATA")
        print(f"  Unseen data file: {unseen_data_path}")
    else:
        print("\n[MODE] Evaluating on VALIDATION and TEST sets")
    
    # Load model and config
    print("\n[STEP 1] Loading trained model and configuration...")
    # Check both possible paths for backward compatibility
    results_dir = project_root / 'results_baseline'
    if not results_dir.exists():
        results_dir = project_root / 'results' / 'baseline'
    
    if not (results_dir / 'classifier.pkl').exists():
        print(f"Error: Classifier not found at {results_dir / 'classifier.pkl'}")
        print("Please run train_baseline.py first!")
        return 1
    
    # Load classifier
    classifier = nb_module.NaiveBayesClassifier(2, 10, 2)
    classifier.load(str(results_dir / 'classifier.pkl'))
    
    # Load config
    with open(results_dir / 'training_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    selected_feature_indices = config['selected_feature_indices']
    optimal_thresholds = config['optimal_thresholds']
    label_to_id = config['label_to_id']
    id_to_label = config['id_to_label']
    min_frames = config.get('min_frames', 120)
    activity_threshold = config.get('activity_threshold', 20.0)
    confidence_threshold = config.get('confidence_threshold', 2.0)
    
    print(f"  [OK] Loaded classifier: {config['num_classes']} classes, {config['num_features']} features")
    print(f"  [OK] Selected features: {selected_feature_indices}")
    print(f"  [OK] Activity threshold: {activity_threshold:.6f}")
    print(f"  [OK] Confidence threshold: {confidence_threshold:.2f}")
    print(f"  [OK] Min frames: {min_frames}")
    
    # Initialize processor
    print("\n[STEP 2] Initializing video processor...")
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    processor.style_classifier = classifier
    
    # Load data - either unseen data or validation/test sets
    if is_unseen_evaluation:
        print("\n[STEP 3] Loading unseen data...")
        unseen_df = pd.read_csv(unseen_data_path)
        # Filter to supported classes (baseline only supports walking and hand_wave_side)
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side'])].copy()
        
        if len(unseen_df) == 0:
            print("Error: No unseen videos found with supported labels (walking, hand_wave_side)!")
            return 1
        
        print(f"  [OK] Loaded {len(unseen_df)} unseen videos")
        
        # For unseen data, we'll evaluate on it as a single "Unseen" set
        val_df = unseen_df
        test_df = pd.DataFrame()  # Empty test set
        set_name = "Unseen Data"
    else:
        # Load validation and test data
        print("\n[STEP 3] Loading validation and test datasets...")
        val_df = pd.read_csv(project_root / 'data' / 'metadata' / 'val_labels.csv')
        val_df = val_df[val_df['label'].isin(['walking', 'hand_wave_side'])].copy()
        
        test_df = pd.read_csv(project_root / 'data' / 'metadata' / 'test_labels.csv')
        test_df = test_df[test_df['label'].isin(['walking', 'hand_wave_side'])].copy()
        
        if len(val_df) == 0:
            print("Error: No validation videos found!")
            return 1
        
        print(f"  [OK] Loaded {len(val_df)} validation videos")
        print(f"  [OK] Loaded {len(test_df)} test videos")
        set_name = "Validation"
    
    print(f"\n[STEP 3A] Evaluating on {set_name} set...")
    val_video_paths = val_df['video_path'].tolist()
    val_labels = val_df['label'].tolist()
    
    val_results = evaluate_on_set(
        processor, val_video_paths, val_labels, label_to_id, set_name=set_name,
        feature_min=config.get('feature_min'),
        feature_max=config.get('feature_max'),
        min_frames=min_frames,
        selected_feature_indices=selected_feature_indices,
        optimal_thresholds=optimal_thresholds,
        activity_threshold=activity_threshold,
        confidence_threshold=confidence_threshold
    )
    
    # Evaluate on test set (only if not unseen evaluation)
    if not is_unseen_evaluation and len(test_df) > 0:
        print("\n[STEP 3B] Evaluating on Test set...")
        test_video_paths = test_df['video_path'].tolist()
        test_labels = test_df['label'].tolist()
        
        test_results = evaluate_on_set(
            processor, test_video_paths, test_labels, label_to_id, set_name="Test",
            feature_min=config.get('feature_min'),
            feature_max=config.get('feature_max'),
            min_frames=min_frames,
            selected_feature_indices=selected_feature_indices,
            optimal_thresholds=optimal_thresholds,
            activity_threshold=activity_threshold,
            confidence_threshold=confidence_threshold
        )
    else:
        test_results = {}
    
    # Combine results for overall metrics
    all_video_stats = []
    if 'video_stats' in val_results:
        all_video_stats.extend(val_results['video_stats'])
    if 'video_stats' in test_results:
        all_video_stats.extend(test_results['video_stats'])
    
    # Use validation results for main metrics (or combined if needed)
    results = val_results
    
    # Extract results
    all_block_predictions = results.get('block_predictions', [])
    all_block_labels = results.get('block_labels', [])
    all_confidence_ratios = results.get('confidence_ratios', [])
    video_level_predictions = results.get('video_predictions', [])
    video_level_labels = results.get('video_labels', [])
    video_stats = results.get('video_stats', [])
    video_predictions_list = results.get('video_predictions_list', [])  # For per-video breakdown
    block_acc = results.get('block_accuracy', 0.0)
    frame_acc = results.get('frame_accuracy', 0.0)
    video_acc = results.get('video_accuracy', 0.0)
    total_blocks = results.get('total_blocks', 0)
    classified_blocks = results.get('classified_blocks', 0)
    unclassified_blocks = results.get('unclassified_blocks', 0)
    
    # Get detailed metrics from results
    num_correct_blocks = results.get('num_correct_blocks', sum(1 for p, l in zip(all_block_predictions, all_block_labels) if p == l))
    num_classified_blocks = results.get('num_classified_blocks', len([p for p in all_block_predictions if p >= 0]))
    num_correct_frames = results.get('num_correct_frames', 0)
    num_total_frames = results.get('num_total_frames', 0)
    video_correct = results.get('video_correct', sum(1 for p, l in zip(video_level_predictions, video_level_labels) if p == l))
    
    # Results already computed by evaluate_on_set
    results_title = "Unseen Data Results Summary" if is_unseen_evaluation else "Validation Set Results Summary"
    print(f"\n" + "=" * 60)
    print(f"{results_title}")
    print("=" * 60)
    print(f"  Block-level accuracy: {block_acc:.2%} ({num_correct_blocks}/{num_classified_blocks} blocks)")
    if num_total_frames > 0:
        print(f"  Frame-level accuracy: {frame_acc:.2%} ({num_correct_frames}/{num_total_frames} frames)")
    else:
        print(f"  Frame-level accuracy: {frame_acc:.2%}")
    print(f"  Video-level accuracy: {video_acc:.2%} ({video_correct}/{len(video_level_predictions)} videos)")
    print(f"  Total blocks: {total_blocks:,}")
    if total_blocks > 0:
        print(f"  Classified blocks: {classified_blocks:,} ({100*classified_blocks/total_blocks:.1f}%)")
        print(f"  Unclassified blocks: {unclassified_blocks:,} ({100*unclassified_blocks/total_blocks:.1f}%)")
    else:
        print(f"  Classified blocks: {classified_blocks:,} (0.0%)")
        print(f"  Unclassified blocks: {unclassified_blocks:,} (0.0%)")
    
    # Print test set results if available
    if len(test_results) > 0:
        test_block_acc = test_results.get('block_accuracy', 0.0)
        test_frame_acc = test_results.get('frame_accuracy', 0.0)
        test_video_acc = test_results.get('video_accuracy', 0.0)
        test_total_blocks = test_results.get('total_blocks', 0)
        test_classified_blocks = test_results.get('classified_blocks', 0)
        test_unclassified_blocks = test_results.get('unclassified_blocks', 0)
        test_video_correct = sum(1 for p, l in zip(test_results.get('video_predictions', []), test_results.get('video_labels', [])) if p == l)
        test_num_correct_blocks = sum(1 for p, l in zip(test_results.get('block_predictions', []), test_results.get('block_labels', [])) if p == l)
        
        print(f"\n" + "=" * 60)
        print(f"Test Set Results Summary")
        print("=" * 60)
        print(f"  Block-level accuracy: {test_block_acc:.2%} ({test_num_correct_blocks}/{test_classified_blocks} blocks)")
        print(f"  Frame-level accuracy: {test_frame_acc:.2%}")
        print(f"  Video-level accuracy: {test_video_acc:.2%} ({test_video_correct}/{len(test_results.get('video_predictions', []))} videos)")
        print(f"  Total blocks: {test_total_blocks:,}")
        print(f"  Classified blocks: {test_classified_blocks:,} ({100*test_classified_blocks/test_total_blocks:.1f}%)")
        print(f"  Unclassified blocks: {test_unclassified_blocks:,} ({100*test_unclassified_blocks/test_total_blocks:.1f}%)")
    
    # Per-class metrics
    if len(video_level_predictions) > 0:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            video_level_labels, video_level_predictions, labels=list(range(config['num_classes'])), zero_division=0
        )
        metrics_data = []
        for i in range(config['num_classes']):
            label_name = id_to_label.get(i, f"class_{i}")
            metrics_data.append({
                'Class': label_name,
                'Precision': precision[i],
                'Recall': recall[i],
                'F1-Score': f1[i],
                'Support': support[i]
            })
        metrics_df = pd.DataFrame(metrics_data)
        print(f"\nPer-class metrics:")
        print(metrics_df)
    
    # Confusion matrix
    if len(video_level_predictions) > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(video_level_labels, video_level_predictions, 
                             labels=list(range(config['num_classes'])))
        print(f"\nConfusion Matrix:")
        print(cm)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Use existing plot generation functions
    results_dict = {
        'block_accuracy': block_acc,
        'frame_accuracy': frame_acc,
        'video_accuracy': video_acc,
        'block_predictions': np.array(all_block_predictions) if len(all_block_predictions) > 0 else np.array([]),
        'block_labels': np.array(all_block_labels) if len(all_block_labels) > 0 else np.array([]),
        'video_predictions': np.array(video_level_predictions) if len(video_level_predictions) > 0 else np.array([]),
        'video_labels': np.array(video_level_labels) if len(video_level_labels) > 0 else np.array([]),
        'confidence_ratios': np.array(all_confidence_ratios) if len(all_confidence_ratios) > 0 else np.array([]),
        'video_stats': video_stats,
        'total_blocks': total_blocks,
        'classified_blocks': classified_blocks,
        'unclassified_blocks': unclassified_blocks
    }
    
    if not is_unseen_evaluation:
        plot_module.generate_all_plots(results_dict, label_to_id, plots_dir,
                                       confidence_ratios=results_dict['confidence_ratios'],
                                       video_stats=video_stats)
    
    # Save CSV files
    print("\n" + "=" * 60)
    print("Saving CSV Files")
    print("=" * 60)
    
    # Per-video breakdown (combine val and test)
    print("\n" + "=" * 60)
    print("Per-Video Breakdown")
    print("=" * 60)
    if all_video_stats:
        video_df = pd.DataFrame(all_video_stats)
        video_df.to_csv(results_dir / 'per_video_breakdown.csv', index=False)
        print(f"  Saved: {results_dir / 'per_video_breakdown.csv'}")
        print(f"\n  Showing per-video statistics:")
        print(video_df.to_string(index=False))
    elif video_stats:
        video_df = pd.DataFrame(video_stats)
        video_df.to_csv(results_dir / 'per_video_breakdown.csv', index=False)
        print(f"  Saved: {results_dir / 'per_video_breakdown.csv'}")
        print(f"\n  Showing per-video statistics:")
        print(video_df.to_string(index=False))
    
    # Per-class metrics
    if len(video_level_predictions) > 0 and 'metrics_df' in locals():
        metrics_df.to_csv(results_dir / 'per_class_metrics.csv', index=False)
        print(f"Saved: {results_dir / 'per_class_metrics.csv'}")
        
        # Confusion matrix CSV
        cm_df = pd.DataFrame(cm, 
                            index=[id_to_label[i] for i in range(config['num_classes'])],
                            columns=[id_to_label[i] for i in range(config['num_classes'])])
        cm_df.to_csv(results_dir / 'confusion_matrix_detailed.csv')
        print(f"Saved: {results_dir / 'confusion_matrix_detailed.csv'}")
    
    # Save metrics summary and training config
    print("\n" + "=" * 60)
    print("Saving Metrics Summary and Config")
    print("=" * 60)
    
    # Save metrics summary manually to include test set results
    summary_path = results_dir / 'metrics_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Evaluation Metrics Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("VALIDATION SET:\n")
        f.write("Overall Accuracy:\n")
        f.write(f"  Block-level: {block_acc:.2%}\n")
        f.write(f"  Frame-level: {frame_acc:.2%}\n")
        f.write(f"  Video-level: {video_acc:.2%}\n\n")
        f.write("Block Statistics:\n")
        f.write(f"  Total blocks: {total_blocks:,}\n")
        if total_blocks > 0:
            f.write(f"  Classified blocks: {classified_blocks:,} ({100*classified_blocks/total_blocks:.2f}%)\n")
            f.write(f"  Unclassified blocks: {unclassified_blocks:,} ({100*unclassified_blocks/total_blocks:.2f}%)\n\n")
        else:
            f.write(f"  Classified blocks: {classified_blocks:,} (0.00%)\n")
            f.write(f"  Unclassified blocks: {unclassified_blocks:,} (0.00%)\n\n")
        f.write("Per-Class Metrics:\n")
        if len(video_level_predictions) > 0:
            for i in range(config['num_classes']):
                label_name = id_to_label.get(i, f"class_{i}")
                f.write(f"  {label_name}:\n")
                f.write(f"    Precision: {precision[i]:.2%}\n")
                f.write(f"    Recall: {recall[i]:.2%}\n")
                f.write(f"    F1-Score: {f1[i]:.2%}\n")
                f.write(f"    Support: {support[i]}\n\n")
        
        # Add test set results if available
        if len(test_results) > 0:
            test_block_acc = test_results.get('block_accuracy', 0.0)
            test_frame_acc = test_results.get('frame_accuracy', 0.0)
            test_video_acc = test_results.get('video_accuracy', 0.0)
            test_total_blocks = test_results.get('total_blocks', 0)
            test_classified_blocks = test_results.get('classified_blocks', 0)
            test_unclassified_blocks = test_results.get('unclassified_blocks', 0)
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("TEST SET:\n")
            f.write("Overall Accuracy:\n")
            f.write(f"  Block-level: {test_block_acc:.2%}\n")
            f.write(f"  Frame-level: {test_frame_acc:.2%}\n")
            f.write(f"  Video-level: {test_video_acc:.2%}\n\n")
            f.write("Block Statistics:\n")
            f.write(f"  Total blocks: {test_total_blocks:,}\n")
            f.write(f"  Classified blocks: {test_classified_blocks:,} ({100*test_classified_blocks/test_total_blocks:.2f}%)\n")
            f.write(f"  Unclassified blocks: {test_unclassified_blocks:,} ({100*test_unclassified_blocks/test_total_blocks:.2f}%)\n\n")
    
    if hasattr(plot_module, 'save_training_config'):
        plot_module.save_training_config(config, results_dir / 'training_config.txt')
    
    print("\nEvaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
