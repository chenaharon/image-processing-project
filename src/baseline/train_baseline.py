"""
Baseline Training Pipeline
Implements Keren (2003) methodology exactly:
- MI-based feature selection
- Threshold optimization for binarization
- Bernoulli Naive Bayes training

2 classes: WAVE_SIDE, WALKING
Non-overlapping blocks (stride=5)
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
sys.path.insert(0, str(project_root / 'src' / 'utils'))
sys.path.insert(0, str(project_root / 'code'))

# Import utils
from naive_bayes import NaiveBayesClassifier

# Import video processor
code_dir = project_root / 'code'
spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)
VideoProcessor = video_processor_module.VideoProcessor


def main():
    """Main training function."""
    print("=" * 60)
    print("BASELINE PIPELINE - Training (Keren 2003 Methodology)")
    print("=" * 60)
    print("\nStarting training process...")
    print("Methodology: Exact reproduction of Keren 2003 paper")
    print("  - Resolution: 64x64")
    print("  - Blocks: 5x5x5 non-overlapping (stride=5)")
    print("  - Features: 10 DCT coefficients")
    print("  - Classifier: Bernoulli Naive Bayes")
    
    # Load training data
    print("\n[STEP 0] Loading training data...")
    print("  Reading train_labels.csv...")
    train_df = pd.read_csv(project_root / 'data' / 'metadata' / 'train_labels.csv')
    print(f"  [OK] Total videos in dataset: {len(train_df)}")
    
    # Filter to WAVE_SIDE and WALKING (baseline: 2 classes)
    print("\n[STEP 0.1] Filtering to WAVE_SIDE and WALKING categories...")
    train_df = train_df[train_df['label'].isin(['walking', 'hand_wave_side'])].copy()
    
    if len(train_df) == 0:
        print("  [ERROR] No videos found!")
        return 1
    
    print(f"  [OK] Loaded {len(train_df)} training videos")
    
    # Show distribution
    print("\n  Video distribution by class:")
    for label, count in train_df['label'].value_counts().items():
        print(f"    {label}: {count} videos")
    
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
    
    # Initialize processor
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    # Find minimum video length
    print("\n" + "=" * 60)
    print("[STEP 1] Finding minimum video length")
    print("=" * 60)
    print("  Checking video lengths...")
    
    video_lengths = []
    for idx, video_path in enumerate(train_df['video_path'], 1):
        if Path(video_path).exists():
            try:
                frames = processor.load_video(video_path, max_frames=None)
                video_lengths.append(len(frames))
                if idx % 5 == 0 or idx == len(train_df):
                    print(f"    Checked {idx}/{len(train_df)} videos...", end='\r')
            except Exception as e:
                print(f"    [WARNING] Could not load {Path(video_path).name}: {e}")
    
    print()  # New line after progress
    if not video_lengths:
        print("  [ERROR] No videos could be loaded!")
        return 1
    
    min_frames = min(video_lengths)
    max_frames = max(video_lengths)
    avg_frames = int(np.mean(video_lengths))
    print(f"  [OK] Video length statistics:")
    print(f"    Minimum: {min_frames} frames")
    print(f"    Maximum: {max_frames} frames")
    print(f"    Average: {avg_frames} frames")
    print(f"  All videos will be trimmed to {min_frames} frames")
    
    # Extract features
    print("\n" + "=" * 60)
    print("[STEP 2] Extracting features from videos")
    print("=" * 60)
    print("  Processing videos and extracting 5x5x5 blocks...")
    
    all_features = []
    all_labels = []
    all_activities = []  # Collect activities for filtering
    
    video_paths = train_df['video_path'].tolist()
    labels = train_df['label'].tolist()
    label_ids = [label_to_id[label] for label in labels]
    
    for idx, (video_path, label_id) in enumerate(zip(video_paths, label_ids), 1):
        if idx % 2 == 0 or idx == len(video_paths):
            print(f"    Processed {idx}/{len(video_paths)} videos...", end="\r")
        if not Path(video_path).exists():
            continue
        
        try:
            # Extract features with activities (paper: filter low-activity blocks)
            features, valid_mask, activities = processor.extract_features_for_training(
                video_path, label_id, max_frames=min_frames, start_from_center=True,
                return_activities=True  # Get activities for filtering
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
    
    # Apply activity filtering (paper: "Blocks with a small time derivative... are not considered")
    if len(activities_flat) > 0:
        # Use percentile-based threshold (similar to paper's approach)
        # Paper doesn't specify exact threshold, but suggests filtering low-activity blocks
        activity_threshold = np.percentile(activities_flat, 10)  # Filter bottom 10% (low activity)
        print(f"\n[STEP 2.5] Activity Filtering (Paper Methodology)")
        print(f"  Activity threshold (10th percentile): {activity_threshold:.6f}")
        print(f"  Total blocks before filtering: {len(X)}")
        
        activity_mask = activities_flat >= activity_threshold
        X = X[activity_mask]
        y = y[activity_mask]
        
        blocks_after = len(X)
        blocks_discarded = len(activities_flat) - blocks_after
        print(f"  Low-activity blocks discarded: {blocks_discarded} ({blocks_discarded/len(activities_flat)*100:.1f}%)")
        print(f"  Active blocks retained: {blocks_after} ({blocks_after/len(activities_flat)*100:.1f}%)")
    else:
        activity_threshold = None
        print("\n[WARNING] No activity data collected, skipping activity filtering")
    
    print(f"\n  [OK] Extracted {len(X)} features from {len(train_df)} videos")
    print(f"  Feature shape: {X.shape}")
    
    # Feature Selection and Binarization (Paper Methodology)
    print("\n" + "=" * 60)
    print("[STEP 3] Feature Selection and Binarization (Paper Methodology)")
    print("=" * 60)
    print("  [3.1] Loading feature extraction module...")
    
    # Import feature extraction module
    fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
    fe_module = importlib.util.module_from_spec(fe_spec)
    sys.modules["feature_extraction"] = fe_module
    fe_spec.loader.exec_module(fe_module)
    
    print("  [3.2] Quantizing features for MI computation...")
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
    
    # Find optimal thresholds for binarization
    optimal_thresholds = fe_module.find_optimal_thresholds_by_mi(X_selected, y, num_candidates=100)
    
    print(f"\nOptimal thresholds:")
    for f_idx, orig_idx in enumerate(selected_feature_indices):
        print(f"  Feature {orig_idx}: threshold = {optimal_thresholds[f_idx]:.4f}")
    
    # Binarize features
    X_binary = fe_module.binarize_features_by_threshold(X_selected, optimal_thresholds)
    
    print(f"\nBinary features shape: {X_binary.shape}")
    print(f"Feature range: [{X_binary.min()}, {X_binary.max()}]")
    
    # Train Bernoulli Naive Bayes
    print("\n" + "=" * 60)
    print("[STEP 4] Training Bernoulli Naive Bayes")
    print("=" * 60)
    print("  Initializing classifier...")
    
    num_classes = len(unique_labels)
    num_features = X_binary.shape[1]
    classifier = NaiveBayesClassifier(num_classes=num_classes, num_features=num_features, num_bins=2)
    
    print("  Training classifier (P(C_i) = n_i/n)...")
    classifier.fit(X_binary, y, balanced_priors=False)  # Paper: P(C_i) = n_i/n
    
    print(f"\n  [OK] Trained classifier on {len(X_binary)} samples")
    print(f"  Class priors: {classifier.class_priors}")
    
    # Save model and config
    print("\n" + "=" * 60)
    print("[STEP 5] Saving Model and Configuration")
    print("=" * 60)
    print("  Saving classifier and configuration files...")
    
    # Check both possible paths for backward compatibility
    results_dir = project_root / 'results_baseline'
    if not results_dir.exists():
        results_dir = project_root / 'results' / 'baseline'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    classifier.save(str(results_dir / 'classifier.pkl'))
    print(f"  [OK] Saved classifier: {results_dir / 'classifier.pkl'}")
    
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
        'stride': 5,
        'activity_threshold': activity_threshold if activity_threshold is not None else 20.0,
        'confidence_threshold': 2.0,
        'feature_min': feature_min,
        'feature_max': feature_max
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
