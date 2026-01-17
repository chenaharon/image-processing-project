"""
Predict activity type for a new video using the trained classifier.
Baseline implementation matching the paper methodology.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse

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


def main():
    parser = argparse.ArgumentParser(description='Predict activity type for a video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--classifier', type=str, default='results/classifier.pkl',
                       help='Path to classifier file')
    parser.add_argument('--mapping', type=str, default='results/label_mapping.pkl',
                       help='Path to label mapping file')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not Path(args.classifier).exists():
        print(f"Error: Classifier not found: {args.classifier}")
        print("Please run train_classifier.py first.")
        return 1
    
    print("=" * 60)
    print("Video Activity Prediction (Baseline - Paper Methodology)")
    print("=" * 60)
    
    # Load training configuration to get parameters
    config_path = 'results/training_config.pkl'
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
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=num_bins)
    processor.style_classifier = nb_module.NaiveBayesClassifier(num_classes, num_features, num_bins)
    processor.style_classifier.load(args.classifier)
    
    # Load label mapping
    with open(args.mapping, 'rb') as f:
        label_to_id = pickle.load(f)
    
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
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    print(f"\nVideo: {Path(args.video).name}")
    if min_frames:
        print(f"Video will be trimmed to {min_frames} frames (matching training)")
    print("Extracting features...")
    
    # Extract features (trim to min_frames if provided, matching training)
    # Extract from center to match training (paper doesn't specify, this is reasonable)
    max_frames = min_frames if min_frames else None
    features, valid_mask, _ = processor.extract_features_for_training(
        args.video, 0, 
        max_frames=max_frames,
        start_from_center=True,  # Extract from center (matching training)
        min_activity=None,  # Activity filtering handled separately if needed
        return_activities=False  # Don't need activities for prediction
    )
    
    if len(features) == 0:
        print("Error: No features extracted from video")
        return 1
    
    # Apply feature selection and binarization (if used in training)
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
    
    # Predict
    probabilities = processor.style_classifier.predict_proba(X_quantized)
    
    # Apply confidence-ratio based classification (Keren 2003 evaluation heuristic)
    # For each block: compute ratio r = max(P_0, P_1) / min(P_0, P_1)
    # If r < threshold → mark as unclassified
    confidence_threshold = 2.0
    max_probs = np.max(probabilities, axis=1)
    min_probs = np.min(probabilities, axis=1)
    confidence_ratios = max_probs / (min_probs + 1e-10)  # r = max/min
    high_confidence_mask = confidence_ratios >= confidence_threshold
    
    if np.sum(high_confidence_mask) > 0:
        predictions = processor.style_classifier.predict(X_quantized[high_confidence_mask])
        probabilities_filtered = probabilities[high_confidence_mask]
        print(f"High-confidence blocks: {np.sum(high_confidence_mask)}/{len(X_quantized)} ({np.sum(high_confidence_mask)/len(X_quantized)*100:.1f}%)")
    else:
        # Fallback if no high-confidence blocks
        predictions = processor.style_classifier.predict(X_quantized)
        probabilities_filtered = probabilities
        print(f"Warning: No high-confidence blocks found. Using all blocks.")
    
    # Video-level prediction (majority vote on high-confidence blocks)
    if len(predictions) > 0:
        video_pred = np.bincount(predictions).argmax()
    else:
        video_pred = 0
    
    avg_prob = probabilities_filtered.mean(axis=0) if len(probabilities_filtered) > 0 else probabilities.mean(axis=0)
    confidence = avg_prob[video_pred] if len(avg_prob) > video_pred else 0.5
    
    predicted_label = id_to_label.get(video_pred, "unknown")
    
    print(f"\nPrediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll class probabilities:")
    for class_id, prob in enumerate(avg_prob):
        label = id_to_label.get(class_id, "unknown")
        print(f"  {label}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
