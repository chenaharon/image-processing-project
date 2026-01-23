"""
Improved Evaluation Pipeline
Evaluates trained classifier with confidence threshold optimization.
Uses same pipeline as baseline but with optimized parameters.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import importlib.util
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.max_open_warning'] = 0

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'code'))

# Import modules
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

sys.path.insert(0, str(project_root / 'src' / 'utils'))
from naive_bayes import NaiveBayesClassifier

# Import metrics utilities
sys.path.insert(0, str(project_root / 'src' / 'utils'))
from metrics import (
    compute_block_level_accuracy,
    compute_frame_level_accuracy,
    compute_video_level_accuracy
)

# Constants
ACTIVITY_THRESHOLD = 20.0  # Variance threshold for activity filtering
CONFIDENCE_THRESHOLD = 2.0  # Confidence ratio threshold

# Additional imports for evaluation functions
from collections import defaultdict
from scipy.fftpack import dct
import cv2
from typing import List, Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def _extract_and_classify_blocks(
    frames: List[np.ndarray],
    processor,
    selected_feature_indices: Optional[np.ndarray],
    optimal_thresholds: Optional[np.ndarray],
    stride: int = 5,
    activity_threshold: float = ACTIVITY_THRESHOLD
) -> Tuple[List, List, List, List, List]:
    """
    Extract blocks, filter by activity, classify, and return results.
    Same logic as in visualize_improved.py - processes frames resized to target_resolution (128x128).
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
    activity_threshold: float = ACTIVITY_THRESHOLD
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
                print("  SKIP (no frames)", flush=True)
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
            classified_predictions = [p for p in all_predictions if p >= 0]
            classified_confidences = [c for p, c in zip(all_predictions, all_confidence_ratios) if p >= 0]
            
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
            
            # Calculate per-video accuracy
            correct_blocks = sum(1 for p, l in zip(classified_predictions, [label_id]*len(classified_predictions)) if p == l)
            video_block_acc = correct_blocks / classified_blocks_video if classified_blocks_video > 0 else 0.0
            
            video_stats.append({
                'video_name': video_path_obj.name,
                'true_label': label_str,
                'total_blocks': total_blocks_video,
                'classified_blocks': classified_blocks_video,
                'unclassified_blocks': unclassified_blocks_video,
                'classified_percentage': 100.0 * classified_blocks_video / total_blocks_video if total_blocks_video > 0 else 0.0
            })
            
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
        'video_correct': video_correct,
        'num_correct_blocks': num_correct_blocks,
        'num_classified_blocks': num_classified_blocks,
        'num_correct_frames': num_correct_frames,
        'num_total_frames': num_total_frames,
        'video_correct': video_correct
    }


def main():
    """Main evaluation function with confidence optimization."""
    import os
    
    print("=" * 60)
    print("IMPROVED PIPELINE - Evaluation")
    print("=" * 60)
    print("\nStarting evaluation process...")
    
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
    results_dir = project_root / 'results_improved'
    if not results_dir.exists():
        print("Error: results_improved directory not found!")
        print("Please run train_improved.py first.")
        return 1
    
    classifier_path = results_dir / 'classifier.pkl'
    config_path = results_dir / 'training_config.pkl'
    label_mapping_path = results_dir / 'label_mapping.pkl'
    
    if not classifier_path.exists() or not config_path.exists():
        print("Error: Model files not found!")
        return 1
    
    # Load config first to get num_classes and num_features
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Create classifier instance and load
    classifier = NaiveBayesClassifier(
        num_classes=config['num_classes'],
        num_features=config['num_features'],
        num_bins=2
    )
    classifier.load(str(classifier_path))
    
    with open(label_mapping_path, 'rb') as f:
        label_to_id = pickle.load(f)
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    selected_feature_indices = config['selected_feature_indices']
    optimal_thresholds = config['optimal_thresholds']
    min_frames = config.get('min_frames', 120)
    activity_threshold = config.get('activity_threshold', ACTIVITY_THRESHOLD)
    confidence_threshold = config.get('confidence_threshold', 2.0)
    
    print(f"  [OK] Loaded classifier: {config['num_classes']} classes, {config['num_features']} features")
    print(f"  [OK] Selected features: {selected_feature_indices}")
    print(f"  [OK] Activity threshold: {activity_threshold:.6f}")
    print(f"  [OK] Confidence threshold: {confidence_threshold:.2f}")
    print(f"  [OK] Min frames: {min_frames}")
    
    # Initialize processor with higher resolution (128x128) matching training
    print("\n[STEP 2] Initializing video processor...")
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32, target_resolution=128)
    processor.style_classifier = classifier
    
    # Load data - either unseen data or validation/test sets
    if is_unseen_evaluation:
        print("\n[STEP 3] Loading unseen data...")
        unseen_df = pd.read_csv(unseen_data_path)
        # Filter to supported classes (improved only supports walking and hand_wave_side)
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side'])].copy()
        
        if len(unseen_df) == 0:
            print("Error: No unseen videos found with supported labels (walking, hand_wave_side)!")
            return 1
        
        print(f"  [OK] Loaded {len(unseen_df)} unseen videos")
        
        # For unseen data, we'll evaluate on it as a single "Unseen" set
        val_df = unseen_df
        test_df = pd.DataFrame()  # Empty test set
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
    
    # Evaluate on validation set (or unseen data)
    if is_unseen_evaluation:
        set_name = "Unseen Data"
    else:
        set_name = "Validation"
    
    print(f"\n[STEP 3A] Evaluating on {set_name} set...")
    val_video_paths = val_df['video_path'].tolist()
    val_labels = val_df['label'].tolist()
    
    val_results = evaluate_on_set(
        processor, val_video_paths, val_labels, label_to_id, set_name=set_name,
        feature_min=None, feature_max=None, min_frames=min_frames,
        selected_feature_indices=selected_feature_indices,
        optimal_thresholds=optimal_thresholds,
        confidence_threshold=confidence_threshold,
        activity_threshold=activity_threshold
    )
    
    # Evaluate on test set (only if not unseen evaluation)
    if not is_unseen_evaluation and len(test_df) > 0:
        print("\n[STEP 3B] Evaluating on Test set...")
        test_video_paths = test_df['video_path'].tolist()
        test_labels = test_df['label'].tolist()
        
        test_results = evaluate_on_set(
            processor, test_video_paths, test_labels, label_to_id, set_name="Test",
            feature_min=None, feature_max=None, min_frames=min_frames,
            selected_feature_indices=selected_feature_indices,
            optimal_thresholds=optimal_thresholds,
            confidence_threshold=confidence_threshold,
            activity_threshold=activity_threshold
        )
    else:
        test_results = {}
    
    # Use validation results for main metrics
    results = val_results
    
    # Combine video stats
    all_video_stats = []
    if 'video_stats' in val_results:
        all_video_stats.extend(val_results['video_stats'])
    if 'video_stats' in test_results:
        all_video_stats.extend(test_results['video_stats'])
    
    # Extract results
    all_block_predictions = results.get('block_predictions', [])
    all_block_labels = results.get('block_labels', [])
    all_confidence_ratios = results.get('confidence_ratios', [])
    video_level_predictions = results.get('video_predictions', [])
    video_level_labels = results.get('video_labels', [])
    video_stats = results.get('video_stats', [])
    video_predictions_list = results.get('video_predictions_list', [])
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
        
        print(f"\nTest Set Results:")
        print(f"  Block-level accuracy: {test_block_acc:.2%}")
        print(f"  Frame-level accuracy: {test_frame_acc:.2%}")
        print(f"  Video-level accuracy: {test_video_acc:.2%}")
        print(f"  Total blocks: {test_total_blocks:,}")
        print(f"  Classified blocks: {test_classified_blocks:,} ({100*test_classified_blocks/test_total_blocks:.1f}%)")
        print(f"  Unclassified blocks: {test_unclassified_blocks:,} ({100*test_unclassified_blocks/test_total_blocks:.1f}%)")
    
    # Per-class metrics
    if len(video_level_predictions) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            video_level_labels, video_level_predictions, 
            labels=list(range(config['num_classes'])), zero_division=0
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
            print(f"\n{label_name}:")
            print(f"  Precision: {precision[i]:.2%}")
            print(f"  Recall: {recall[i]:.2%}")
            print(f"  F1-Score: {f1[i]:.2%}")
            print(f"  Support: {support[i]}")
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(results_dir / 'per_class_metrics.csv', index=False)
    
    # Confusion matrix
    if len(video_level_predictions) > 0:
        cm = confusion_matrix(video_level_labels, video_level_predictions, 
                            labels=list(range(config['num_classes'])))
        cm_df = pd.DataFrame(cm, 
                            index=[id_to_label.get(i, f"class_{i}") for i in range(config['num_classes'])],
                            columns=[id_to_label.get(i, f"class_{i}") for i in range(config['num_classes'])])
        cm_df.to_csv(results_dir / 'confusion_matrix.csv', index=True)
        
        # Detailed confusion matrix
        cm_detailed = []
        for i in range(config['num_classes']):
            for j in range(config['num_classes']):
                cm_detailed.append({
                    'True_Label': id_to_label.get(i, f"class_{i}"),
                    'Predicted_Label': id_to_label.get(j, f"class_{j}"),
                    'Count': int(cm[i, j])
                })
        cm_detailed_df = pd.DataFrame(cm_detailed)
        cm_detailed_df.to_csv(results_dir / 'confusion_matrix_detailed.csv', index=False)
    
    # Per-video breakdown (combine val and test)
    if len(all_video_stats) > 0:
        breakdown_data = []
        for stat in all_video_stats:
            breakdown_data.append({
                'video_name': stat.get('video_name', 'unknown'),
                'true_label': stat.get('true_label', 'unknown'),
                'predicted_label': stat.get('predicted_label', 'unknown'),
                'correct': stat.get('correct', False),
                'confidence': stat.get('confidence', 0.0),
                'total_blocks': stat.get('total_blocks', 0),
                'classified_blocks': stat.get('classified_blocks', 0),
                'unclassified_blocks': stat.get('unclassified_blocks', 0),
                'classified_percentage': stat.get('classified_percentage', 0.0),
                'hand_wave_side_blocks': stat.get('hand_wave_side_blocks', 0),
                'walking_blocks': stat.get('walking_blocks', 0)
            })
        breakdown_df = pd.DataFrame(breakdown_data)
        breakdown_df.to_csv(results_dir / 'per_video_breakdown.csv', index=False)
        print(f"  Saved: {results_dir / 'per_video_breakdown.csv'}")
        print(f"\n  Showing per-video statistics:")
        print(breakdown_df.to_string(index=False))
    
    # Generate plots (skip for unseen data to save time)
    if not is_unseen_evaluation:
        print("\n" + "=" * 60)
        print("Generating Plots")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Skipping Plots Generation (Unseen Data Mode)")
        print("=" * 60)
    
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    if not is_unseen_evaluation:
        # 1. Accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        accuracies = [block_acc, frame_acc, video_acc]
        labels_plot = ['Block-level', 'Frame-level', 'Video-level']
        bars = ax.bar(labels_plot, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] accuracy_comparison.png")
        
        # 2. Confusion matrix
        if len(video_level_predictions) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(plots_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] confusion_matrix.png")
        
        # 3. Per-class metrics
        if len(video_level_predictions) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(metrics_df))
            width = 0.25
            ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#3498db')
            ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
            ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#e74c3c')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['Class'])
            ax.legend()
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(plots_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] per_class_metrics.png")
        
        # 4. Unclassified blocks pie chart
        if total_blocks > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            sizes = [classified_blocks, unclassified_blocks]
            labels_pie = ['Classified', 'Unclassified']
            colors = ['#2ecc71', '#95a5a6']
            ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Classified vs Unclassified Blocks', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / 'unclassified_blocks_pie.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] unclassified_blocks_pie.png")
        
        # 5. Confidence distribution
        if len(all_confidence_ratios) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_confidence_ratios, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Confidence Threshold (R={confidence_threshold})')
            ax.set_xlabel('Confidence Ratio (max_prob / min_prob)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Confidence Ratio Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] confidence_distribution.png")
        
        # 6. Block distribution per video
        if len(breakdown_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            video_names = breakdown_df['video_name'].tolist()
            hand_wave_side_counts = breakdown_df['hand_wave_side_blocks'].tolist()
            walking_counts = breakdown_df['walking_blocks'].tolist()
            x = np.arange(len(video_names))
            width = 0.35
            ax.bar(x - width/2, hand_wave_side_counts, width, label='hand_wave_side', color='#f39c12')
            ax.bar(x + width/2, walking_counts, width, label='walking', color='#9b59b6')
            ax.set_xlabel('Video', fontsize=12)
            ax.set_ylabel('Number of Blocks', fontsize=12)
            ax.set_title('Block Distribution per Video', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(video_names, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'block_distribution_per_video.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  [OK] block_distribution_per_video.png")
    
    # Save updated config
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    # Save metrics summary
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
        f.write(f"  Classified blocks: {classified_blocks:,} ({100*classified_blocks/total_blocks:.2f}%)\n")
        f.write(f"  Unclassified blocks: {unclassified_blocks:,} ({100*unclassified_blocks/total_blocks:.2f}%)\n\n")
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
        
        f.write(f"\nConfidence Threshold: R = {confidence_threshold}\n")
    
    # Save training config text
    config_txt_path = results_dir / 'training_config.txt'
    with open(config_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write("Block Parameters:\n")
        f.write(f"  Block size: {config.get('block_size', 'N/A')}\n")
        f.write(f"  Stride: {config.get('stride', 'N/A')}\n")
        f.write(f"  Temporal window: {config.get('temporal_window', 'N/A')}\n")
        f.write(f"  Spatial resolution: {config.get('spatial_resolution', 'N/A')}\n\n")
        f.write("Feature Parameters:\n")
        f.write(f"  Selected features: {config.get('num_features_selected', config.get('num_features', 'N/A'))}\n")
        f.write(f"  Activity threshold: {config.get('activity_threshold', 'N/A')}\n")
        f.write(f"  Activity percentile: {config.get('activity_percentile', 'N/A')}\n")
        f.write(f"  Confidence threshold: {config.get('confidence_threshold', 'N/A')}\n")
        f.write(f"  Confidence optimized: {config.get('confidence_threshold_optimized', False)}\n\n")
        f.write("Training Parameters:\n")
        f.write(f"  Number of classes: {config.get('num_classes', 'N/A')}\n")
        f.write(f"  Min frames: {config.get('min_frames', 'N/A')}\n\n")
        f.write("Selected Feature Indices:\n")
        f.write(f"  {config.get('selected_feature_indices', [])}\n\n")
        f.write("Optimal Thresholds:\n")
        thresholds = config.get('optimal_thresholds', [])
        for i, thresh in enumerate(thresholds[:10]):  # Show first 10
            f.write(f"  Feature {i}: {thresh:.4f}\n")
        if len(thresholds) > 10:
            f.write(f"  ... and {len(thresholds) - 10} more\n")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
