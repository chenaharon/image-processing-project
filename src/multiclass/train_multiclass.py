"""
Multiclass Training Pipeline
3-class classifier: HELLO, WAVE_SIDE, WALKING
Uses same methodology as baseline but with 3 classes.
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
nb_spec = importlib.util.spec_from_file_location("naive_bayes_classifier", code_dir / "naive_bayes_classifier.py")
nb_module = importlib.util.module_from_spec(nb_spec)
sys.modules["naive_bayes_classifier"] = nb_module
nb_spec.loader.exec_module(nb_module)

# Import utils
sys.path.insert(0, str(project_root / 'src' / 'utils'))
from naive_bayes import NaiveBayesClassifier


def main():
    """Main training function."""
    print("=" * 60)
    print("MULTICLASS PIPELINE - Training (3 classes)")
    print("=" * 60)
    print("\nClasses: HELLO, WAVE_SIDE, WALKING")
    print("\nStarting training process...")
    
    # Load training data
    print("\n[STEP 0] Loading training data...")
    train_df = pd.read_csv(project_root / 'data' / 'metadata' / 'train_labels.csv')
    print(f"  Total videos in dataset: {len(train_df)}")
    
    # Filter to 3 classes
    print("\n[STEP 0.1] Filtering to HELLO, WAVE_SIDE, and WALKING categories...")
    train_df = train_df[train_df['label'].isin(['hand_wave_hello', 'hand_wave_side', 'walking'])].copy()
    
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
    
    # Configuration (same as IMPROVED)
    TARGET_RESOLUTION = 128  # Higher resolution (same as IMPROVED)
    NUM_THRESHOLD_CANDIDATES = 200  # More candidates (same as IMPROVED)
    ACTIVITY_PERCENTILE = 10  # Activity filtering (same as IMPROVED)
    
    # Initialize processor with higher resolution (128x128) matching IMPROVED
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32, target_resolution=TARGET_RESOLUTION)
    
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
    
    # Extract features (non-overlapping, stride=5)
    print("\n" + "=" * 60)
    print("Step 2: Extracting features from videos")
    print("=" * 60)
    
    all_features = []
    all_labels = []
    all_activities = []  # For percentile-based filtering (same as IMPROVED)
    
    video_paths = train_df['video_path'].tolist()
    labels = train_df['label'].tolist()
    label_ids = [label_to_id[label] for label in labels]
    
    for idx, (video_path, label_id) in enumerate(zip(video_paths, label_ids), 1):
        if idx % 2 == 0 or idx == len(video_paths):
            print(f"    Processed {idx}/{len(video_paths)} videos...", end="\r")
        if not Path(video_path).exists():
            continue
        
        try:
            # Extract features with activities (same as IMPROVED)
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
    
    # Smart activity filtering (percentile-based, same as IMPROVED)
    print("\n" + "=" * 60)
    print("Step 2.5: Smart Activity Filtering (Percentile-based)")
    print("=" * 60)
    
    if len(activities_flat) > 0:
        activity_threshold = np.percentile(activities_flat, ACTIVITY_PERCENTILE)
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
        print("Warning: No activity data available, skipping activity filtering")
    
    # Feature Selection and Binarization (same as baseline)
    print("\n" + "=" * 60)
    print("Step 3: Feature Selection and Binarization")
    print("=" * 60)
    
    # Quantize features for MI computation
    X_quantized_temp, feature_min, feature_max = fe_module.quantize_features(X, processor.num_bins)
    
    # Compute mutual information
    mi_matrix = fe_module.compute_mutual_information(X_quantized_temp, y, num_bins=processor.num_bins)
    avg_mi = np.mean(mi_matrix, axis=0)
    
    # Select top 10 features
    top_k = 10
    selected_feature_indices = np.argsort(avg_mi)[-top_k:][::-1]
    
    print(f"Selected {len(selected_feature_indices)} features based on MI:")
    for idx in selected_feature_indices:
        print(f"  Feature {idx}: MI = {avg_mi[idx]:.4f}")
    
    # Use only selected features
    X_selected = X[:, selected_feature_indices]
    
    # Find optimal thresholds for binarization (200 candidates, same as IMPROVED)
    optimal_thresholds = fe_module.find_optimal_thresholds_by_mi(X_selected, y, num_candidates=NUM_THRESHOLD_CANDIDATES)
    
    print(f"\nOptimal thresholds:")
    for f_idx, orig_idx in enumerate(selected_feature_indices):
        print(f"  Feature {orig_idx}: threshold = {optimal_thresholds[f_idx]:.4f}")
    
    # Binarize features
    X_binary = fe_module.binarize_features_by_threshold(X_selected, optimal_thresholds)
    
    print(f"\nBinary features shape: {X_binary.shape}")
    print(f"Feature range: [{X_binary.min()}, {X_binary.max()}]")
    
    # Train Bernoulli Naive Bayes (3 classes)
    print("\n" + "=" * 60)
    print("Step 4: Training Bernoulli Naive Bayes (3 classes)")
    print("=" * 60)
    
    num_classes = len(unique_labels)  # 3 classes
    num_features = X_binary.shape[1]
    classifier = NaiveBayesClassifier(num_classes=num_classes, num_features=num_features, num_bins=2)
    classifier.fit(X_binary, y, balanced_priors=False)  # Paper: P(C_i) = n_i/n
    
    print(f"\nTrained classifier on {len(X_binary)} samples")
    print(f"Class priors: {classifier.class_priors}")
    
    # Save model and config
    results_dir = project_root / 'results_multiclass'
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
        'stride': 5,  # Non-overlapping (same as IMPROVED)
        'activity_percentile': ACTIVITY_PERCENTILE,  # Same as IMPROVED
        'confidence_threshold': 2.0,
        'spatial_resolution': TARGET_RESOLUTION,  # 128x128 (same as IMPROVED)
        'feature_min': feature_min,
        'feature_max': feature_max,
        'min_frames': min_frames
    }
    
    with open(results_dir / 'training_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    with open(results_dir / 'label_mapping.pkl', 'wb') as f:
        pickle.dump(label_to_id, f)
    
    print(f"\nModel saved to: {results_dir}")
    print("Training complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
