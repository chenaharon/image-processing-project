"""
Improved Training Pipeline - Quick Improvements to Baseline
Fast improvements that should improve accuracy without much slowdown:
1. More features: 15 (instead of 10) - quick improvement
2. Smart activity filtering: percentile-based (65th percentile) - quick
3. Confidence threshold optimization: R in {1.5, 2.0, 2.5, 3.0} - done in evaluation
4. Same resolution (64x64) - keeps speed
5. Same non-overlapping blocks - keeps speed
6. More candidate thresholds for binarization (200 instead of 100) - quick
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'code'))

# Import video processor
code_dir = project_root / 'code'
spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)
VideoProcessor = video_processor_module.VideoProcessor

# Import feature extraction
fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
fe_module = importlib.util.module_from_spec(fe_spec)
sys.modules["feature_extraction"] = fe_module
fe_spec.loader.exec_module(fe_module)

# Import naive bayes
sys.path.insert(0, str(project_root / 'src' / 'utils'))
from naive_bayes import NaiveBayesClassifier


def compute_percentile_threshold(activities, percentile=65):
    """Compute activity threshold as percentile of distribution."""
    return np.percentile(activities, percentile)


def main():
    """Main training function with quick improvements."""
    print("=" * 60)
    print("IMPROVED PIPELINE - Training (Quick Improvements)")
    print("=" * 60)
    print("\nIMPROVEMENTS OVER BASELINE (fast, should improve accuracy):")
    print("  1. Higher resolution: 128x128 (instead of 64x64) - better spatial detail")
    print("  2. Activity filtering: 10th percentile (removes static blocks, like baseline)")
    print("  3. Confidence threshold optimization: R in {1.5, 2.0, 2.5, 3.0}")
    print("  4. More candidate thresholds: 200 (instead of 100 baseline)")
    print("  5. Same non-overlapping blocks - keeps speed")
    print("\nStarting training process...")
    
    # Configuration
    NUM_FEATURES = 10  # Use all 10 DCT coefficients (we only extract 10)
    ACTIVITY_PERCENTILE = 10  # Improvement: Filter bottom 10% (like baseline, removes static blocks)
    NUM_THRESHOLD_CANDIDATES = 200  # Improvement: More candidates for binarization (200 vs 100 baseline)
    
    # Load training data
    print("\n[STEP 0] Loading training data...")
    train_df = pd.read_csv(project_root / 'data' / 'metadata' / 'train_labels.csv')
    print(f"  Total videos in dataset: {len(train_df)}")
    
    # Filter to WAVE_SIDE and WALKING
    print("\n[STEP 0.1] Filtering to WAVE_SIDE and WALKING categories...")
    train_df = train_df[train_df['label'].isin(['walking', 'hand_wave_side'])].copy()
    
    if len(train_df) == 0:
        print("Error: No videos found!")
        return 1
    
    print(f"Loaded {len(train_df)} training videos")
    
    # Balance videos per class
    min_videos = train_df['label'].value_counts().min()
    balanced_videos = []
    for label in sorted(train_df['label'].unique()):
        label_videos = train_df[train_df['label'] == label].copy()
        if len(label_videos) > min_videos:
            np.random.seed(42)
            label_videos = label_videos.sample(n=min_videos, random_state=42)
        balanced_videos.append(label_videos)
    train_df = pd.concat(balanced_videos, ignore_index=True)
    
    print(f"\nAfter balancing:")
    for label, count in train_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # Create label mapping
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"\nLabel mapping: {label_to_id}")
    
    # Initialize processor with higher resolution (128x128) for improved pipeline
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32, target_resolution=128)
    
    # Find minimum video length
    print("\n" + "=" * 60)
    print("Step 1: Finding minimum video length")
    print("=" * 60)
    
    video_lengths = []
    for video_path in train_df['video_path']:
        if Path(video_path).exists():
            try:
                frames = processor.load_video(video_path, max_frames=None)
                video_lengths.append(len(frames))
            except:
                pass
    
    if not video_lengths:
        print("Error: No videos could be loaded!")
        return 1
    
    min_frames = min(video_lengths)
    print(f"Minimum video length: {min_frames} frames")
    print(f"All videos will be trimmed to {min_frames} frames")
    
    # Extract features
    print("\n" + "=" * 60)
    print("Step 2: Extracting features from videos")
    print("=" * 60)
    
    all_features = []
    all_labels = []
    all_activities = []  # For percentile-based filtering
    
    video_paths = train_df['video_path'].tolist()
    labels = train_df['label'].tolist()
    label_ids = [label_to_id[label] for label in labels]
    
    for idx, (video_path, label_id) in enumerate(zip(video_paths, label_ids), 1):
        if idx % 2 == 0 or idx == len(video_paths):
            print(f"    Processed {idx}/{len(video_paths)} videos...", end="\r")
        if not Path(video_path).exists():
            continue
        
        try:
            # Extract features with activities
            features, valid_mask, activities = processor.extract_features_for_training(
                video_path, label_id, max_frames=min_frames, start_from_center=True, return_activities=True
            )
            all_features.append(features)
            if activities is not None:
                all_activities.append(activities)
            all_labels.extend([label_id] * len(features))
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    print()  # newline after progress
    
    X = np.vstack(all_features)  # Shape: (N, 10) - 10 DCT coefficients
    activities_flat = np.concatenate(all_activities) if all_activities else np.array([])
    y = np.array(all_labels)
    
    print(f"\nExtracted {len(X)} features from {len(train_df)} videos")
    
    # Smart activity filtering (percentile-based)
    print("\n" + "=" * 60)
    print("Step 2.5: Smart Activity Filtering (Percentile-based)")
    print("=" * 60)
    
    if len(activities_flat) > 0:
        activity_threshold = compute_percentile_threshold(activities_flat, percentile=ACTIVITY_PERCENTILE)
        print(f"Activity threshold ({ACTIVITY_PERCENTILE}th percentile): {activity_threshold:.6f}")
        print(f"  Total blocks before filtering: {len(X)}")
        
        # Filter blocks by activity (paper: "Blocks with a small time derivative... are not considered")
        active_mask = activities_flat >= activity_threshold
        X_active = X[active_mask]
        y_active = y[active_mask]
        
        blocks_after = len(X_active)
        blocks_discarded = len(X) - blocks_after
        print(f"  Low-activity blocks discarded: {blocks_discarded} ({blocks_discarded/len(X)*100:.1f}%)")
        print(f"  Active blocks retained: {blocks_after} ({blocks_after/len(X)*100:.1f}%)")
        X = X_active
        y = y_active
    else:
        activity_threshold = 20.0  # Fallback
        print("Warning: No activity data available, using hardcoded threshold 20.0")
    
    # Feature Selection and Binarization (with more features)
    print("\n" + "=" * 60)
    print(f"Step 3: Feature Selection and Binarization ({NUM_FEATURES} features)")
    print("=" * 60)
    
    # Quantize features for MI computation
    X_quantized_temp, feature_min, feature_max = fe_module.quantize_features(X, processor.num_bins)
    
    # Compute mutual information
    mi_matrix = fe_module.compute_mutual_information(X_quantized_temp, y, num_bins=processor.num_bins)
    avg_mi = np.mean(mi_matrix, axis=0)
    
    # Select top NUM_FEATURES features (we have 10 DCT coefficients, so we'll use all 10)
    max_available_features = min(NUM_FEATURES, X.shape[1])
    selected_feature_indices = np.argsort(avg_mi)[-max_available_features:][::-1]
    
    print(f"Selected {len(selected_feature_indices)} features based on MI:")
    for idx in selected_feature_indices:
        print(f"  Feature {idx}: MI = {avg_mi[idx]:.4f}")
    
    # Use only selected features
    X_selected = X[:, selected_feature_indices]
    
    # Find optimal thresholds for binarization (with more candidates)
    optimal_thresholds = fe_module.find_optimal_thresholds_by_mi(X_selected, y, num_candidates=NUM_THRESHOLD_CANDIDATES)
    
    print(f"\nOptimal thresholds (first 5):")
    for f_idx, orig_idx in enumerate(selected_feature_indices[:5]):
        print(f"  Feature {orig_idx}: threshold = {optimal_thresholds[f_idx]:.4f}")
    
    # Binarize features
    X_binary = fe_module.binarize_features_by_threshold(X_selected, optimal_thresholds)
    
    print(f"\nBinary features shape: {X_binary.shape}")
    print(f"Feature range: [{X_binary.min()}, {X_binary.max()}]")
    
    # Train Bernoulli Naive Bayes
    print("\n" + "=" * 60)
    print("Step 4: Training Bernoulli Naive Bayes")
    print("=" * 60)
    
    num_classes = len(unique_labels)
    num_features = X_binary.shape[1]
    classifier = NaiveBayesClassifier(num_classes=num_classes, num_features=num_features, num_bins=2)
    classifier.fit(X_binary, y, balanced_priors=False)  # Paper: P(C_i) = n_i/n
    
    print(f"\nTrained classifier on {len(X_binary)} samples")
    print(f"Class priors: {classifier.class_priors}")
    
    # Save model and config
    results_dir = project_root / 'results_improved'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    classifier.save(str(results_dir / 'classifier.pkl'))
    
    # Save config
    config = {
        'selected_feature_indices': selected_feature_indices,
        'optimal_thresholds': optimal_thresholds,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_classes': num_classes,
        'num_features': num_features,
        'block_size': 5,
        'temporal_window': 5,
        'stride': 5,  # Non-overlapping
        'spatial_resolution': 128,  # Higher resolution for improved pipeline
        'activity_threshold': activity_threshold,  # Percentile-based
        'activity_percentile': ACTIVITY_PERCENTILE,
        'activity_threshold': activity_threshold,
        'confidence_threshold': 2.0,  # Will be optimized in evaluation
        'confidence_candidates': [1.5, 2.0, 2.5, 3.0],
        'feature_min': feature_min,
        'feature_max': feature_max,
        'min_frames': min_frames,
        'num_features_selected': len(selected_feature_indices)
    }
    
    with open(results_dir / 'training_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    with open(results_dir / 'label_mapping.pkl', 'wb') as f:
        pickle.dump(label_to_id, f)
    
    print(f"\nModel saved to: {results_dir}")
    print("Training complete!")
    print("\n" + "=" * 60)
    print("IMPROVEMENTS SUMMARY:")
    print("=" * 60)
    print(f"  [OK] Features: {len(selected_feature_indices)} (same as baseline - all 10 DCT coefficients)")
    print(f"  [OK] Activity filtering: {ACTIVITY_PERCENTILE}th percentile = {activity_threshold:.6f} (removes static blocks)")
    print(f"  [OK] Threshold candidates: {NUM_THRESHOLD_CANDIDATES} (vs 100 baseline)")
    print(f"  [OK] Confidence optimization: R in {config['confidence_candidates']}")
    print(f"  [OK] Resolution: 128x128 (vs 64x64 baseline - better spatial detail)")
    print(f"  [OK] Blocks: Non-overlapping (same as baseline - keeps speed)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
