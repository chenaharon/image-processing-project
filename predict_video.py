"""
Predict activity type for a new video using the trained classifier.
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
    print("Video Activity Prediction")
    print("=" * 60)
    
    # Load classifier
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    processor.style_classifier = nb_module.NaiveBayesClassifier(3, 10, 32)
    processor.style_classifier.load(args.classifier)
    
    # Load label mapping
    with open(args.mapping, 'rb') as f:
        label_to_id = pickle.load(f)
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    print(f"\nVideo: {Path(args.video).name}")
    print("Extracting features...")
    
    # Extract features
    features = processor.extract_features_for_training(args.video, 0, max_frames=90)
    
    if len(features) == 0:
        print("Error: No features extracted from video")
        return 1
    
    # Quantize features
    X_quantized = fe_module.quantize_features(features, processor.num_bins)
    
    # Predict
    predictions = processor.style_classifier.predict(X_quantized)
    probabilities = processor.style_classifier.predict_proba(X_quantized)
    
    # Video-level prediction (majority vote)
    video_pred = np.bincount(predictions).argmax()
    avg_prob = probabilities.mean(axis=0)
    confidence = avg_prob[video_pred]
    
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

