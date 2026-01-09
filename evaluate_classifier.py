"""
Evaluate the trained classifier on validation and test sets.
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


def evaluate_on_set(processor, video_paths, labels, label_to_id, set_name="Test"):
    """Evaluate classifier on a set of videos."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {set_name} Set")
    print(f"{'='*60}")
    
    all_predictions = []
    all_true_labels = []
    video_level_predictions = []
    video_level_labels = []
    
    for video_path, true_label in zip(video_paths, labels):
        print(f"Processing: {Path(video_path).name}...", end=' ', flush=True)
        
        try:
            # Extract features
            features = processor.extract_features_for_training(video_path, 0, max_frames=90)
            
            if len(features) == 0:
                print("Skipped (no features)")
                continue
            
            # Quantize features
            X_quantized = fe_module.quantize_features(features, processor.num_bins)
            
            # Predict
            predictions = processor.style_classifier.predict(X_quantized)
            probabilities = processor.style_classifier.predict_proba(X_quantized)
            
            # Block-level predictions
            all_predictions.extend(predictions)
            label_id = label_to_id[true_label]
            all_true_labels.extend([label_id] * len(predictions))
            
            # Video-level prediction (majority vote)
            video_pred = np.bincount(predictions).argmax()
            video_level_predictions.append(video_pred)
            video_level_labels.append(label_id)
            
            # Get prediction confidence
            avg_prob = probabilities.mean(axis=0)
            predicted_class = np.argmax(avg_prob)
            confidence = avg_prob[predicted_class]
            
            # Get class name
            id_to_label = {v: k for k, v in label_to_id.items()}
            predicted_label = id_to_label.get(predicted_class, "unknown")
            
            status = "✓" if video_pred == label_id else "✗"
            print(f"{status} Predicted: {predicted_label} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    video_level_predictions = np.array(video_level_predictions)
    video_level_labels = np.array(video_level_labels)
    
    # Block-level accuracy
    block_accuracy = np.mean(all_predictions == all_true_labels)
    
    # Video-level accuracy
    video_accuracy = np.mean(video_level_predictions == video_level_labels)
    
    print(f"\n{set_name} Set Results:")
    print(f"  Block-level accuracy: {block_accuracy:.2%}")
    print(f"  Video-level accuracy: {video_accuracy:.2%}")
    print(f"  Total blocks: {len(all_predictions)}")
    print(f"  Total videos: {len(video_level_predictions)}")
    
    return {
        'block_accuracy': block_accuracy,
        'video_accuracy': video_accuracy,
        'block_predictions': all_predictions,
        'block_labels': all_true_labels,
        'video_predictions': video_level_predictions,
        'video_labels': video_level_labels
    }


def main():
    print("=" * 60)
    print("Classifier Evaluation")
    print("=" * 60)
    
    # Load classifier
    classifier_path = 'results/classifier.pkl'
    mapping_path = 'results/label_mapping.pkl'
    
    if not Path(classifier_path).exists():
        print(f"Error: Classifier not found at {classifier_path}")
        print("Please run train_classifier.py first.")
        return 1
    
    # Initialize processor
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    # Load classifier
    processor.style_classifier = nb_module.NaiveBayesClassifier(3, 10, 32)
    processor.style_classifier.load(classifier_path)
    
    # Load label mapping
    with open(mapping_path, 'rb') as f:
        label_to_id = pickle.load(f)
    
    print(f"\nLoaded classifier from {classifier_path}")
    print(f"Label mapping: {label_to_id}")
    
    # Evaluate on validation set
    val_df = pd.read_csv('data/metadata/val_labels.csv')
    if len(val_df) > 0:
        val_results = evaluate_on_set(
            processor,
            val_df['video_path'].tolist(),
            val_df['label'].tolist(),
            label_to_id,
            "Validation"
        )
    
    # Evaluate on test set
    test_df = pd.read_csv('data/metadata/test_labels.csv')
    if len(test_df) > 0:
        test_results = evaluate_on_set(
            processor,
            test_df['video_path'].tolist(),
            test_df['label'].tolist(),
            label_to_id,
            "Test"
        )
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())

