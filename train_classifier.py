"""
Train the style/motion classifier on the dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
    print("Training Style/Motion Classifier")
    print("=" * 60)
    
    # Load training data
    train_df = pd.read_csv('data/metadata/train_labels.csv')
    
    print(f"\nLoaded {len(train_df)} training videos")
    print(f"Categories: {train_df['label'].unique()}")
    print(f"\nLabel distribution:")
    for label, count in train_df['label'].value_counts().items():
        print(f"  {label}: {count}")
    
    # Get video paths and labels
    video_paths = train_df['video_path'].tolist()
    labels = train_df['label'].tolist()
    
    # Convert labels to numeric IDs
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = [label_to_id[label] for label in labels]
    
    print(f"\nLabel mapping:")
    for label, idx in label_to_id.items():
        print(f"  {label} -> {idx}")
    
    # Initialize processor with paper parameters: 5x5 blocks, 5 frame temporal window
    print("\nInitializing video processor...")
    print("Using paper methodology: 5x5 spatial blocks, 5-frame temporal window, 64x64 resolution")
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    # Limit frames per video (need at least 5 for temporal window)
    # For 2-5 second videos at 30fps: 60-150 frames
    MAX_FRAMES_PER_VIDEO = 90  # ~3 seconds at 30fps
    
    # Train classifier
    print(f"\nTraining classifier (max {MAX_FRAMES_PER_VIDEO} frames per video)...")
    print("Extracting features using 5x5x5 spatio-temporal neighborhoods (as in paper)...")
    
    try:
        all_features = []
        all_labels = []
        
        for video_path, label in zip(video_paths, label_ids):
            print(f"Processing: {Path(video_path).name}...", end=' ', flush=True)
            features = processor.extract_features_for_training(
                video_path, 
                label, 
                max_frames=MAX_FRAMES_PER_VIDEO
            )
            print(f"Done ({len(features)} features)")
            all_features.append(features)
            all_labels.extend([label] * len(features))
        
        # Concatenate all features
        X = np.vstack(all_features)
        y = np.array(all_labels)
        
        # Import needed modules
        import importlib.util
        
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
        
        # Quantize features
        X_quantized = fe_module.quantize_features(X, processor.num_bins)
        
        # Train classifier
        num_classes = len(np.unique(label_ids))
        num_features = X_quantized.shape[1]
        
        processor.style_classifier = nb_module.NaiveBayesClassifier(num_classes, num_features, processor.num_bins)
        processor.style_classifier.fit(X_quantized, y)
        
        print(f"\n[OK] Trained classifier on {len(X)} samples, {num_classes} classes")
        
        # Save the classifier
        classifier_path = 'results/classifier.pkl'
        Path('results').mkdir(exist_ok=True)
        processor.style_classifier.save(classifier_path)
        print(f"\n[OK] Classifier saved to {classifier_path}")
        
        # Save label mapping
        import pickle
        mapping_path = 'results/label_mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(label_to_id, f)
        print(f"[OK] Label mapping saved to {mapping_path}")
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

