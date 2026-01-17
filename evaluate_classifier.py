"""
Evaluate the trained classifier on validation and test sets.
Baseline implementation matching the paper methodology.

Implements Keren (2003) evaluation heuristics:
- Confidence-ratio based unclassified blocks (r = max/min probability ratio)
- Block-level statistics on VAL/TEST (percentage of classified blocks per class)
- Frame-level and video-level aggregation via majority vote
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add code directory to path
code_dir = Path(__file__).parent / 'code'
sys.path.insert(0, str(code_dir))

# Import modules
import importlib.util

# Load video_processor
spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)
VideoProcessor = video_processor_module.VideoProcessor

# Load feature_extraction
fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
fe_module = importlib.util.module_from_spec(fe_spec)
sys.modules["feature_extraction"] = fe_module
fe_spec.loader.exec_module(fe_module)

# Load naive_bayes_classifier
nb_spec = importlib.util.spec_from_file_location("naive_bayes_classifier", code_dir / "naive_bayes_classifier.py")
nb_module = importlib.util.module_from_spec(nb_spec)
sys.modules["naive_bayes_classifier"] = nb_module
nb_spec.loader.exec_module(nb_module)


def evaluate_on_set(processor, video_paths, labels, label_to_id, set_name="Test",
                   feature_min=None, feature_max=None, min_frames=None,
                   selected_feature_indices=None, optimal_thresholds=None):
    """Evaluate classifier on a set of videos."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {set_name} Set")
    print(f"{'='*60}")
    
    all_predictions = []
    all_true_labels = []
    video_level_predictions = []
    video_level_labels = []
    video_predictions_list = []  # Store predictions per video for distribution calculation
    
    total_videos = len(video_paths)
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    for idx, (video_path, true_label) in enumerate(zip(video_paths, labels), 1):
        video_name = Path(video_path).name
        print(f"[{idx}/{total_videos}] {video_name}...", end=' ', flush=True)
        
        try:
            # Extract features (trim to min_frames if provided, matching training)
            # Extract from center to match training (paper doesn't specify, this is reasonable)
            max_frames = min_frames if min_frames else None
            features, valid_mask, _ = processor.extract_features_for_training(
                video_path, 0, 
                max_frames=max_frames,
                start_from_center=True,  # Extract from center (matching training)
                min_activity=None,  # Activity filtering handled separately if needed
                return_activities=False  # Don't need activities for evaluation
            )
            
            if len(features) == 0:
                print("Skipped (no features)")
                continue
            
            # Apply feature selection and binarization (reuse exact pipeline from training)
            if selected_feature_indices is not None and optimal_thresholds is not None:
                # Select features
                features_selected = features[:, selected_feature_indices]
                # Binarize using optimal thresholds
                X_quantized = fe_module.binarize_features_by_threshold(features_selected, optimal_thresholds)
            else:
                # Fallback to quantization (old method)
                X_quantized, _, _ = fe_module.quantize_features(
                    features, 
                    processor.num_bins,
                    feature_min=feature_min,
                    feature_max=feature_max
                )
            
            # Predict using trained Bernoulli-style Naive Bayes model
            # Implements Keren (2003) evaluation: compute posterior probabilities P(C_0|x) and P(C_1|x)
            probabilities = processor.style_classifier.predict_proba(X_quantized)
            
            # Apply confidence-ratio based classification (Keren 2003 evaluation heuristic)
            # For each block: compute ratio r = max(P_0, P_1) / min(P_0, P_1)
            # If r < threshold → mark as unclassified
            confidence_threshold = 2.0  # Configurable threshold (paper suggests ~2)
            max_probs = np.max(probabilities, axis=1)
            min_probs = np.min(probabilities, axis=1)
            confidence_ratios = max_probs / (min_probs + 1e-10)  # r = max/min
            high_confidence_mask = confidence_ratios >= confidence_threshold
            
            # Count blocks discarded due to low confidence
            total_blocks = len(X_quantized)
            low_confidence_blocks = np.sum(~high_confidence_mask)
            high_confidence_blocks = np.sum(high_confidence_mask)
            
            # Only use high-confidence blocks for classification (Keren 2003: mark others as "unclassified")
            if high_confidence_blocks > 0:
                predictions = processor.style_classifier.predict(X_quantized[high_confidence_mask])
                probabilities_filtered = probabilities[high_confidence_mask]
                classified_block_indices = np.where(high_confidence_mask)[0]
            else:
                # If no high-confidence blocks, use all (fallback)
                predictions = processor.style_classifier.predict(X_quantized)
                probabilities_filtered = probabilities
                high_confidence_mask = np.ones(len(X_quantized), dtype=bool)
                low_confidence_blocks = 0
                high_confidence_blocks = total_blocks
                classified_block_indices = np.arange(len(X_quantized))
            
            # Compute frame-level predictions (Keren 2003: aggregate block predictions to frames)
            # Each 5×5×5 block belongs to a frame (center frame of the temporal window)
            # Blocks are extracted with: spatial stride=5, temporal stride=1, temporal_window=5
            # Block extraction order: for each (i, j) spatial position, for each temporal position t
            # Block index = spatial_block_idx * num_temporal_positions + temporal_pos
            # Frame index for block = temporal_pos + 2 (center of 5-frame window, 0-indexed)
            num_frames_used = min_frames if min_frames else 120  # Default from training
            num_spatial_blocks = 12 * 12  # (64-5)/5 + 1 = 12 blocks per dimension, 12*12 = 144 total
            num_temporal_positions = num_frames_used - 5 + 1  # Number of temporal positions
            
            # Map each block to its frame index
            frame_predictions = {}  # frame_idx -> list of predictions
            for pred_idx, block_idx in enumerate(classified_block_indices):
                # Calculate which temporal position this block belongs to
                temporal_pos = block_idx % num_temporal_positions
                frame_idx = temporal_pos + 2  # Center frame of 5-frame window (0-indexed)
                if frame_idx not in frame_predictions:
                    frame_predictions[frame_idx] = []
                frame_predictions[frame_idx].append(predictions[pred_idx])
            
            # Derive frame-level labels by majority vote over classified blocks (Keren 2003)
            frame_level_predictions = []
            frame_level_labels = []
            for frame_idx in sorted(frame_predictions.keys()):
                if len(frame_predictions[frame_idx]) > 0:
                    frame_pred = np.bincount(frame_predictions[frame_idx]).argmax()
                    frame_level_predictions.append(frame_pred)
                    frame_level_labels.append(label_to_id[true_label])
            
            # Store predictions for this video (for per-video distribution calculation)
            # Note: Only high-confidence blocks are included (classified blocks)
            video_predictions_list.append((
                video_name, predictions, true_label, 
                high_confidence_blocks, total_blocks, low_confidence_blocks,
                frame_level_predictions, frame_level_labels
            ))
            
            # Block-level predictions (only high-confidence blocks)
            all_predictions.extend(predictions)
            label_id = label_to_id[true_label]
            all_true_labels.extend([label_id] * len(predictions))
            
            # Video-level prediction (Keren 2003: majority vote over all classified blocks in video)
            if len(predictions) > 0:
                video_pred = np.bincount(predictions).argmax()
            else:
                # Fallback if no high-confidence blocks
                video_pred = processor.style_classifier.predict(X_quantized)[0]
            
            video_level_predictions.append(video_pred)
            video_level_labels.append(label_id)
            
            # Get prediction confidence (from filtered probabilities)
            avg_prob = probabilities_filtered.mean(axis=0) if len(probabilities_filtered) > 0 else probabilities.mean(axis=0)
            confidence = avg_prob[video_pred] if len(avg_prob) > video_pred else 0.5
            
            # Get class name
            predicted_label = id_to_label.get(video_pred, "unknown")
            
            status = "OK" if video_pred == label_id else "X"
            true_label_name = true_label
            print(f"{status} true:{true_label_name} -> pred:{predicted_label} (conf:{confidence:.2f})")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Calculate metrics (Keren 2003 evaluation statistics)
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    video_level_predictions = np.array(video_level_predictions)
    video_level_labels = np.array(video_level_labels)
    
    # Aggregate frame-level predictions for frame-level accuracy
    all_frame_predictions = []
    all_frame_labels = []
    for item in video_predictions_list:
        if len(item) >= 8:
            _, _, _, _, _, _, frame_preds, frame_labels = item
            all_frame_predictions.extend(frame_preds)
            all_frame_labels.extend(frame_labels)
    
    # Block-level accuracy (on classified blocks only)
    block_accuracy = np.mean(all_predictions == all_true_labels) if len(all_predictions) > 0 else 0.0
    
    # Frame-level accuracy (Keren 2003: frame labels by majority vote)
    frame_accuracy = np.mean(np.array(all_frame_predictions) == np.array(all_frame_labels)) if len(all_frame_predictions) > 0 else 0.0
    
    # Video-level accuracy (Keren 2003: video labels by majority vote over classified blocks)
    video_accuracy = np.mean(video_level_predictions == video_level_labels) if len(video_level_predictions) > 0 else 0.0
    
    print(f"\n{set_name} Set Results (Keren 2003 evaluation):")
    print(f"  Block-level accuracy (on classified blocks only): {block_accuracy:.2%}")
    print(f"  Frame-level accuracy: {frame_accuracy:.2%}")
    print(f"  Video-level accuracy: {video_accuracy:.2%}")
    print(f"  Total classified blocks: {len(all_predictions)}")
    print(f"  Total videos: {len(video_level_predictions)}")
    
    # Pixel/Block Distribution per video (Keren 2003 Section 8, Figures 6-7)
    # Paper reports: "Altogether, 83% of the classified pixels were labeled as 'walking'"
    #                 "Altogether, 98% of the classified pixels were labeled as 'hand waving'"
    # Paper: "each pixel gets the label of its central block"
    # Note: Since blocks are 5×5 non-overlapping, block-level distribution = pixel-level distribution
    print(f"\nPixel/Block Distribution per video (Keren 2003 Section 8, Figures 6-7):")
    print(f"Note: Only classified blocks are included (confidence ratio r >= 2.0, unclassified blocks excluded)")
    print(f"      Block labels are equivalent to pixel labels (5×5 non-overlapping blocks)")
    
    total_all_blocks = 0
    total_classified_blocks = 0
    total_low_confidence_blocks = 0
    
    for item in video_predictions_list:
        if len(item) >= 8:
            video_name, predictions, true_label, num_high_conf, total_blocks, low_conf_blocks, frame_preds, frame_labels = item
        elif len(item) == 6:
            video_name, predictions, true_label, num_high_conf, total_blocks, low_conf_blocks = item
            frame_preds, frame_labels = [], []
        elif len(item) == 5:
            video_name, predictions, true_label, num_high_conf, total_blocks = item
            low_conf_blocks = total_blocks - num_high_conf
            frame_preds, frame_labels = [], []
        else:
            # Backward compatibility
            video_name, predictions, true_label = item
            num_high_conf = len(predictions)
            total_blocks = len(predictions)
            low_conf_blocks = 0
            frame_preds, frame_labels = [], []
        
        total_all_blocks += total_blocks
        total_classified_blocks += num_high_conf
        total_low_confidence_blocks += low_conf_blocks
        
        total_blocks_video = len(predictions)  # Classified blocks only
        if total_blocks_video == 0:
            continue
        
        # Count blocks per class for this video (only classified blocks)
        distribution_video = {}
        for class_id in range(len(label_to_id)):
            count = np.sum(predictions == class_id)
            percentage = (count / total_blocks_video * 100) if total_blocks_video > 0 else 0
            label_name = id_to_label.get(class_id, f"class_{class_id}")
            distribution_video[label_name] = (count, percentage)
        
        # Print distribution for this video (matching paper's reporting style)
        print(f"\n  {video_name} (true label: {true_label}):")
        print(f"    Total blocks: {total_blocks}")
        print(f"    Classified blocks: {num_high_conf} ({num_high_conf/total_blocks*100:.1f}%)")
        print(f"    Unclassified blocks (low confidence): {low_conf_blocks} ({low_conf_blocks/total_blocks*100:.1f}%)")
        print(f"    Distribution of classified blocks:")
        for label_name, (count, percentage) in sorted(distribution_video.items()):
            print(f"      {label_name}: {count} blocks ({percentage:.1f}% of classified blocks)")
    
    # Overall distribution (Keren 2003: percentage of classified blocks per class)
    total_classified_blocks = len(all_predictions)
    distribution_overall = {}
    for class_id in range(len(label_to_id)):
        count = np.sum(all_predictions == class_id)
        percentage = (count / total_classified_blocks * 100) if total_classified_blocks > 0 else 0
        label_name = id_to_label.get(class_id, f"class_{class_id}")
        distribution_overall[label_name] = (count, percentage)
    
    print(f"\n  Overall distribution (all videos combined - Keren 2003 style):")
    print(f"    Total blocks: {total_all_blocks}")
    if total_all_blocks > 0:
        print(f"    Unclassified blocks (low confidence): {total_low_confidence_blocks} ({total_low_confidence_blocks/total_all_blocks*100:.1f}%)")
        print(f"    Classified blocks: {total_classified_blocks} ({total_classified_blocks/total_all_blocks*100:.1f}%)")
    else:
        print(f"    Unclassified blocks (low confidence): {total_low_confidence_blocks}")
        print(f"    Classified blocks: {total_classified_blocks}")
    print(f"    Distribution of classified blocks:")
    for label_name, (count, percentage) in sorted(distribution_overall.items()):
        print(f"      {label_name}: {count} blocks ({percentage:.1f}% of classified blocks)")
    
    return {
        'block_accuracy': block_accuracy,
        'frame_accuracy': frame_accuracy,
        'video_accuracy': video_accuracy,
        'block_predictions': all_predictions,
        'block_labels': all_true_labels,
        'frame_predictions': all_frame_predictions,
        'frame_labels': all_frame_labels,
        'video_predictions': video_level_predictions,
        'video_labels': video_level_labels,
        'total_blocks': total_all_blocks,
        'classified_blocks': total_classified_blocks,
        'unclassified_blocks': total_low_confidence_blocks
    }


def main():
    print("=" * 60)
    print("Classifier Evaluation (Baseline - Paper Methodology)")
    print("=" * 60)
    
    # Load classifier
    classifier_path = 'results/classifier.pkl'
    mapping_path = 'results/label_mapping.pkl'
    config_path = 'results/training_config.pkl'
    
    if not Path(classifier_path).exists():
        print(f"Error: Classifier not found at {classifier_path}")
        print("Please run train_classifier.py first.")
        return 1
    
    # Load training configuration to get parameters
    if Path(config_path).exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        min_frames = config.get('min_frames')
        num_features = config.get('num_features', 10)
        num_bins = config.get('num_bins', 2)  # Default to 2 for binary features
        num_classes = config.get('num_classes', 2)
        selected_feature_indices = config.get('selected_feature_indices', None)
        optimal_thresholds = config.get('optimal_thresholds', None)
        use_binary_features = config.get('use_binary_features', False)
        feature_min_config = config.get('feature_min', None)
        feature_max_config = config.get('feature_max', None)
    else:
        print("Warning: Training configuration not found. Using defaults.")
        min_frames = None
        num_features = 10
        num_bins = 32
        num_classes = 2
        selected_feature_indices = None
        optimal_thresholds = None
        use_binary_features = False
        feature_min_config = None
        feature_max_config = None
    
    # Initialize processor (must match training parameters)
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    # Load label mapping
    with open(mapping_path, 'rb') as f:
        label_to_id = pickle.load(f)
    
    # Load classifier with correct dimensions (must match training!)
    processor.style_classifier = nb_module.NaiveBayesClassifier(num_classes, num_features, num_bins)
    processor.style_classifier.load(classifier_path)
    
    # Load feature normalization parameters (for backward compatibility)
    # If using binary features, feature_min/max come from config
    norm_path = 'results/feature_normalization.pkl'
    if Path(norm_path).exists() and not use_binary_features:
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
            feature_min = norm_params['feature_min']
            feature_max = norm_params['feature_max']
    elif feature_min_config is not None and feature_max_config is not None:
        # Use from config (for binary features)
        feature_min = feature_min_config
        feature_max = feature_max_config
    else:
        print("Warning: Feature normalization parameters not found. Using per-video normalization.")
        feature_min = None
        feature_max = None
    
    print(f"\nLoaded classifier from {classifier_path}")
    print(f"Label mapping: {label_to_id}")
    if min_frames:
        print(f"Videos will be trimmed to {min_frames} frames (matching training)")
    
    # Filter to only hand_wave_hello and walking (as in paper)
    print("\nFiltering to hand_wave_hello and walking categories (as in paper)...")
    
    # Evaluate on validation set
    val_df = pd.read_csv('data/metadata/val_labels.csv')
    val_df = val_df[val_df['label'].isin(['hand_wave_hello', 'walking'])].copy()
    if len(val_df) > 0:
        val_results = evaluate_on_set(
            processor,
            val_df['video_path'].tolist(),
            val_df['label'].tolist(),
            label_to_id,
            "Validation",
            feature_min=feature_min,
            feature_max=feature_max,
            min_frames=min_frames,
            selected_feature_indices=selected_feature_indices,
            optimal_thresholds=optimal_thresholds
        )
    
    # Evaluate on test set
    test_df = pd.read_csv('data/metadata/test_labels.csv')
    test_df = test_df[test_df['label'].isin(['hand_wave_hello', 'walking'])].copy()
    if len(test_df) > 0:
        test_results = evaluate_on_set(
            processor,
            test_df['video_path'].tolist(),
            test_df['label'].tolist(),
            label_to_id,
            "Test",
            feature_min=feature_min,
            feature_max=feature_max,
            min_frames=min_frames,
            selected_feature_indices=selected_feature_indices,
            optimal_thresholds=optimal_thresholds
        )
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
