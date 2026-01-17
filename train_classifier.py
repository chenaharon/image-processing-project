"""
Train the style/motion classifier on the dataset.
Baseline implementation matching the paper methodology.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add code directory to path
code_dir = Path(__file__).parent / 'code'
sys.path.insert(0, str(code_dir))

# Import with absolute path
import importlib.util
spec = importlib.util.spec_from_file_location("video_processor", code_dir / "video_processor.py")
video_processor_module = importlib.util.module_from_spec(spec)
sys.modules["video_processor"] = video_processor_module
spec.loader.exec_module(video_processor_module)

VideoProcessor = video_processor_module.VideoProcessor


def main():
    print("=" * 60)
    print("Training Style/Motion Classifier (Baseline - Paper Methodology)")
    print("=" * 60)
    
    # Load training data
    train_df = pd.read_csv('data/metadata/train_labels.csv')
    
    # Filter to only hand_wave_hello and walking (as in paper: walking vs hand waving)
    print("\nFiltering to hand_wave_hello and walking categories (as in paper)...")
    train_df = train_df[train_df['label'].isin(['hand_wave_hello', 'walking'])].copy()
    
    if len(train_df) == 0:
        print("Error: No videos found for hand_wave_hello and walking categories!")
        return 1
    
    print(f"\nLoaded {len(train_df)} training videos (before balancing)")
    print(f"Categories: {sorted(train_df['label'].unique())}")
    print(f"\nLabel distribution (before balancing):")
    for label, count in train_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # ===== STEP 1: Balance number of videos per class =====
    print("\n" + "=" * 60)
    print("Step 1: Balancing number of videos per class")
    print("=" * 60)
    
    min_videos_per_class = train_df['label'].value_counts().min()
    print(f"Minimum videos per class: {min_videos_per_class}")
    
    # Sample equal number of videos from each class
    balanced_videos = []
    for label in sorted(train_df['label'].unique()):
        label_videos = train_df[train_df['label'] == label].copy()
        if len(label_videos) > min_videos_per_class:
            # Random sample to get min_videos_per_class
            np.random.seed(42)  # For reproducibility
            label_videos = label_videos.sample(n=min_videos_per_class, random_state=42)
        balanced_videos.append(label_videos)
    
    train_df = pd.concat(balanced_videos, ignore_index=True)
    
    print(f"\nAfter balancing videos:")
    for label, count in train_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # ===== STEP 2: Filter out non-existent videos =====
    print("\n" + "=" * 60)
    print("Step 2: Filtering out non-existent videos")
    print("=" * 60)
    
    # Filter out videos that don't exist
    existing_videos = []
    for idx, row in train_df.iterrows():
        video_path = row['video_path']
        if Path(video_path).exists():
            existing_videos.append(idx)
        else:
            print(f"Warning: Video not found, skipping: {video_path}")
    
    train_df = train_df.loc[existing_videos].reset_index(drop=True)
    
    if len(train_df) == 0:
        print("Error: No valid videos found!")
        return 1
    
    print(f"\nAfter filtering: {len(train_df)} valid videos")
    
    # Re-balance after filtering
    min_videos_per_class = train_df['label'].value_counts().min()
    print(f"Minimum videos per class (after filtering): {min_videos_per_class}")
    
    balanced_videos = []
    for label in sorted(train_df['label'].unique()):
        label_videos = train_df[train_df['label'] == label].copy()
        if len(label_videos) > min_videos_per_class:
            np.random.seed(42)
            label_videos = label_videos.sample(n=min_videos_per_class, random_state=42)
        balanced_videos.append(label_videos)
    
    train_df = pd.concat(balanced_videos, ignore_index=True)
    
    print(f"\nAfter re-balancing:")
    for label, count in train_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # ===== STEP 3: Find minimum video length =====
    print("\n" + "=" * 60)
    print("Step 3: Finding minimum video length (frames)")
    print("=" * 60)
    
    # Initialize processor with paper parameters
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    video_lengths = []
    print("\nChecking video lengths...")
    for idx, video_path in enumerate(train_df['video_path'], 1):
        video_name = Path(video_path).name
        try:
            frames = processor.load_video(video_path, max_frames=None)
            num_frames = len(frames)
            video_lengths.append((video_path, num_frames))
            print(f"[{idx}/{len(train_df)}] {video_name}: {num_frames} frames")
        except Exception as e:
            print(f"[{idx}/{len(train_df)}] {video_name}: ERROR - {e}")
            continue
    
    if not video_lengths:
        print("Error: No videos could be loaded!")
        return 1
    
    # Filter train_df to only include videos that were successfully loaded
    valid_video_paths = {path for path, _ in video_lengths}
    train_df = train_df[train_df['video_path'].isin(valid_video_paths)].reset_index(drop=True)
    
    min_frames = min(length for _, length in video_lengths)
    print(f"\nMinimum video length: {min_frames} frames")
    print(f"All videos will be trimmed to {min_frames} frames (extracting from center to capture main action)")
    
    # Filter out videos that are too short (less than 5 frames needed for temporal window)
    if min_frames < 5:
        print(f"Error: Minimum video length ({min_frames}) is less than temporal window (5 frames)")
        return 1
    
    # ===== STEP 4: Extract features with balanced video lengths =====
    print("\n" + "=" * 60)
    print("Step 4: Extracting features (all videos trimmed to same length)")
    print("=" * 60)
    
    video_paths = train_df['video_path'].tolist()
    labels = train_df['label'].tolist()
    
    # Convert labels to numeric IDs
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = [label_to_id[label] for label in labels]
    
    print(f"\nLabel mapping:")
    for label, idx in label_to_id.items():
        print(f"  {label} -> {idx}")
    
    print(f"\nUsing paper methodology:")
    print(f"  - 5x5 spatial blocks")
    print(f"  - 5-frame temporal window")
    print(f"  - 64x64 resolution")
    print(f"  - 3D DCT on full 5x5x5 volume (as in paper)")
    print(f"  - 10 DCT coefficients")
    print(f"  - 32 quantization bins")
    print(f"  - All videos trimmed to: {min_frames} frames")
    
    # Temporal activity filtering (paper: "Blocks with a small time derivative... are not considered")
    # Paper: Filter low-activity blocks before training
    # We compute average squared difference between consecutive frames as temporal activity measure
    print("\nCollecting block temporal activities for filtering threshold computation...")
    all_features = []
    all_labels = []
    all_activities = []
    
    total_videos = len(video_paths)
    for idx, (video_path, label) in enumerate(zip(video_paths, label_ids), 1):
        video_name = Path(video_path).name
        
        # Skip if file doesn't exist
        if not Path(video_path).exists():
            print(f"[{idx}/{total_videos}] Skipping {video_name} (file not found)", flush=True)
            continue
        
        print(f"[{idx}/{total_videos}] Processing: {video_name}...", flush=True)
        
        try:
            # Extract features with max_frames=min_frames (all videos same length)
            # Extract from center to capture the main action (paper doesn't specify, this is reasonable)
            # Collect temporal activities for threshold computation
            features, valid_mask, activities = processor.extract_features_for_training(
                video_path, 
                label, 
                max_frames=min_frames,  # All videos trimmed to same length
                start_from_center=True,  # Extract from center (captures main action)
                min_activity=None,  # Will filter after computing threshold
                return_activities=True  # Return activities for threshold computation
            )
            
            # Store features, labels, and activities
            all_features.append(features)
            all_labels.extend([label] * len(features))
            if activities is not None:
                all_activities.append(activities)
            
            print(f"[{idx}/{total_videos}] Done: {video_name} ({len(features)} features)", flush=True)
        except Exception as e:
            print(f"[{idx}/{total_videos}] Error processing {video_name}: {e}", flush=True)
            continue
    
    # Concatenate all features and activities
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    # Compute temporal activity threshold (paper: filter low-activity blocks)
    # Use 10th percentile as threshold (filter bottom 10% of blocks by temporal activity)
    total_blocks_before_filtering = len(X)
    if len(all_activities) > 0:
        all_activities_flat = np.concatenate(all_activities)
        activity_threshold = np.percentile(all_activities_flat, 10)
        print(f"\nTemporal activity threshold (10th percentile): {activity_threshold:.6f}")
        print(f"Total blocks before activity filtering: {total_blocks_before_filtering}")
        
        # Filter low-activity blocks
        activity_mask = all_activities_flat >= activity_threshold
        X = X[activity_mask]
        y = y[activity_mask]
        blocks_after_filtering = len(X)
        blocks_discarded = total_blocks_before_filtering - blocks_after_filtering
        print(f"Low-activity blocks discarded: {blocks_discarded} ({blocks_discarded/total_blocks_before_filtering*100:.1f}%)")
        print(f"Total blocks after activity filtering: {blocks_after_filtering} ({blocks_after_filtering/total_blocks_before_filtering*100:.1f}% retained)")
    else:
        print("\nWarning: No activities collected, skipping activity filtering")
        blocks_discarded = 0
    
    num_classes = len(np.unique(label_ids))
    
    # ===== Verification: Check if everything is balanced =====
    print("\n" + "=" * 60)
    print("Verification: Feature distribution")
    print("=" * 60)
    
    for c in range(num_classes):
        count = np.sum(y == c)
        percentage = count / len(y) * 100
        label_name = [k for k, v in label_to_id.items() if v == c][0]
        print(f"  {label_name}: {count} features ({percentage:.1f}%)")
    
    # Since videos are balanced and same length, features should be balanced
    # No need for BALANCE_FEATURES = True
    
    # ===== Feature Selection and Binarization (Paper Methodology) =====
    print("\n" + "=" * 60)
    print("Step 5: Feature Selection and Binarization (Paper Methodology)")
    print("=" * 60)
    
    # Load feature_extraction module
    fe_spec = importlib.util.spec_from_file_location("feature_extraction", code_dir / "feature_extraction.py")
    fe_module = importlib.util.module_from_spec(fe_spec)
    sys.modules["feature_extraction"] = fe_module
    fe_spec.loader.exec_module(fe_module)
    
    # Paper methodology: MI-based feature selection and binarization
    # Pipeline (as per paper):
    # 1. Extract continuous 3D DCT coefficients (already done)
    # 2. Quantize into bins (for MI computation)
    # 3. Compute mutual information per feature
    # 4. Select top 10 features based on MI
    # 5. For each selected feature, search threshold that maximizes MI
    # 6. Binarize each feature to 0/1 according to that threshold
    print("\nMI-based feature selection and binarization pipeline:")
    print("  1. Quantize continuous features to discrete bins (for MI computation)")
    print("  2. Compute mutual information per feature")
    print("  3. Select top features based on MI")
    print("  4. Find optimal thresholds that maximize MI")
    print("  5. Binarize features (0/1) using optimal thresholds")
    
    # Step 1: Quantize features to compute MI (need discrete values for MI)
    print("\nStep 1: Quantizing features for MI computation...")
    X_quantized_temp, feature_min, feature_max = fe_module.quantize_features(X, processor.num_bins)
    
    # Step 2: Compute mutual information for all features
    print("Step 2: Computing mutual information for all features...")
    mi_matrix = fe_module.compute_mutual_information(X_quantized_temp, y, num_bins=processor.num_bins)
    avg_mi = np.mean(mi_matrix, axis=0)
    
    print(f"Mutual information per feature:")
    for f in range(len(avg_mi)):
        print(f"  Feature {f}: MI = {avg_mi[f]:.4f}")
    
    # Step 3: Select top features based on MI (paper: "choose a few features which have the largest mutual information")
    print("\nStep 3: Selecting top features based on MI...")
    positive_mi_features = np.where(avg_mi > 0)[0]
    if len(positive_mi_features) > 10:
        # Select top 10 features
        top_k = 10
        selected_feature_indices = np.argsort(avg_mi)[-top_k:][::-1]
    else:
        # Use all features with positive MI
        selected_feature_indices = positive_mi_features
    
    print(f"Selected {len(selected_feature_indices)} features based on mutual information:")
    for idx in selected_feature_indices:
        print(f"  Feature {idx}: MI = {avg_mi[idx]:.4f}")
    
    # Use only selected features
    X_selected = X[:, selected_feature_indices]
    
    # Step 4: Find optimal thresholds for binarization that maximize MI
    print("\nStep 4: Finding optimal thresholds for binarization (maximizing MI)...")
    optimal_thresholds = fe_module.find_optimal_thresholds_by_mi(X_selected, y, num_candidates=100)
    
    print(f"Optimal thresholds:")
    for f_idx, orig_idx in enumerate(selected_feature_indices):
        print(f"  Feature {orig_idx}: threshold = {optimal_thresholds[f_idx]:.4f}")
    
    # Step 5: Binarize features using optimal thresholds
    print("\nStep 5: Binarizing features using optimal thresholds...")
    X_binary = fe_module.binarize_features_by_threshold(X_selected, optimal_thresholds)
    
    print(f"Feature binarization complete:")
    print(f"  - Selected features: {len(selected_feature_indices)}")
    print(f"  - Binary feature range: [{X_binary.min()}, {X_binary.max()}]")
    print(f"  - Features are binary (0 or 1) as per paper methodology")
    print(f"  - Ready for Bernoulli-style Naive Bayes training")
    
    # For Naive Bayes, we'll use binary features (2 bins per feature)
    X_quantized = X_binary
    num_bins_per_feature = 2  # Binary features
    
    # ===== Train classifier =====
    print("\n" + "=" * 60)
    print("Step 6: Training Naive Bayes classifier")
    print("=" * 60)
    
    # Load naive_bayes_classifier module
    nb_spec = importlib.util.spec_from_file_location("naive_bayes_classifier", code_dir / "naive_bayes_classifier.py")
    nb_module = importlib.util.module_from_spec(nb_spec)
    sys.modules["naive_bayes_classifier"] = nb_module
    nb_spec.loader.exec_module(nb_module)
    
    num_features = X_quantized.shape[1]
    # Use 2 bins per feature (binary) for selected features
    processor.style_classifier = nb_module.NaiveBayesClassifier(num_classes, num_features, num_bins_per_feature)
    
    # Paper methodology: P(C_i) = n_i/n where n_i is the number of FEATURES of class i
    # Since we balanced videos and video lengths, n_i = n_j for all i,j
    # So P(C_i) = n_i/n will automatically give 0.5/0.5 (balanced priors)
    # We use balanced_priors=False to follow paper's formula, which gives balanced result
    print(f"\nUsing paper methodology: P(C_i) = n_i/n")
    print("Note: Since videos are balanced, priors will be 0.5/0.5 automatically")
    processor.style_classifier.fit(X_quantized, y, balanced_priors=False)
    
    print(f"\n[OK] Trained classifier on {len(X)} samples, {num_classes} classes")
    
    # Display class priors
    print(f"\nClass Priors (P(C_i) = n_i/n):")
    for c in range(num_classes):
        label_name = [k for k, v in label_to_id.items() if v == c][0]
        print(f"  {label_name} (class {c}): {processor.style_classifier.class_priors[c]:.4f}")
    
    # Calculate and display training accuracy (for monitoring, not evaluation)
    print("\nTraining Accuracy (on training data - for monitoring only):")
    train_predictions = processor.style_classifier.predict(X_quantized)
    train_accuracy = np.mean(train_predictions == y)
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print("Note: This is not evaluation - use evaluate_classifier.py for proper evaluation on validation/test sets")
    
    # Per-class training accuracy
    print("\nPer-class Training Accuracy:")
    for c in range(num_classes):
        label_name = [k for k, v in label_to_id.items() if v == c][0]
        class_mask = (y == c)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(train_predictions[class_mask] == y[class_mask])
            correct = np.sum(train_predictions[class_mask] == y[class_mask])
            total = np.sum(class_mask)
            print(f"  {label_name} (class {c}): {class_accuracy:.2%} ({correct}/{total} correct)")
    
    # Pixel/Block Distribution (as reported in paper Section 8)
    # Paper: "each pixel gets the label of its central block"
    # We report block-level distribution (blocks are 5×5 non-overlapping, so each pixel belongs to one block)
    print("\nPixel/Block Distribution (as reported in paper - Section 8):")
    print(f"Note: All training blocks are classified (no confidence filtering during training)")
    total_blocks = len(train_predictions)
    print(f"Total training blocks: {total_blocks}")
    for c in range(num_classes):
        label_name = [k for k, v in label_to_id.items() if v == c][0]
        count = np.sum(train_predictions == c)
        percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
        print(f"  {label_name}: {count} blocks ({percentage:.1f}% of all blocks)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Total features extracted: {len(X)}")
    print(f"  Selected features per sample: {num_features} (from {X.shape[1]} original)")
    print(f"  Feature representation: Binary (0 or 1) - as per paper methodology")
    print(f"  Feature value range: [{X_quantized.min()}, {X_quantized.max()}]")
    
    # Class distribution
    print(f"\nClass Distribution in Training Set:")
    for c in range(num_classes):
        label_name = [k for k, v in label_to_id.items() if v == c][0]
        count = np.sum(y == c)
        percentage = count / len(y) * 100
        print(f"  {label_name} (class {c}): {count} samples ({percentage:.1f}%)")
    
    # ===== Save the classifier =====
    print("\n" + "=" * 60)
    print("Step 7: Saving classifier and parameters")
    print("=" * 60)
    
    classifier_path = 'results/classifier.pkl'
    Path('results').mkdir(exist_ok=True)
    processor.style_classifier.save(classifier_path)
    print(f"[OK] Classifier saved to {classifier_path}")
    
    # Save label mapping
    import pickle
    mapping_path = 'results/label_mapping.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump(label_to_id, f)
    print(f"[OK] Label mapping saved to {mapping_path}")
    
    # Save feature normalization parameters (min/max) for consistent quantization
    norm_path = 'results/feature_normalization.pkl'
    with open(norm_path, 'wb') as f:
        pickle.dump({
            'feature_min': feature_min,
            'feature_max': feature_max
        }, f)
    print(f"[OK] Feature normalization parameters saved to {norm_path}")
    
    # Save training configuration for reference
    config_path = 'results/training_config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump({
            'min_frames': min_frames,
            'num_videos_per_class': min_videos_per_class,
            'num_classes': num_classes,
            'num_features': num_features,
            'num_bins': num_bins_per_feature,  # 2 for binary features
            'block_size': processor.block_size,
            'num_coefficients': processor.num_coefficients,
            'temporal_window': 5,
            'selected_feature_indices': selected_feature_indices,  # For feature selection
            'optimal_thresholds': optimal_thresholds,  # For binarization
            'use_binary_features': True,  # Flag to indicate binary features
            'feature_min': feature_min,  # For feature selection during evaluation
            'feature_max': feature_max
        }, f)
    print(f"[OK] Training configuration saved to {config_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
