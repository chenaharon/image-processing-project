"""
Basic test script to verify the implementation works correctly.
Run this to test individual components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from feature_extraction import extract_dct_features, quantize_features
from naive_bayes_classifier import NaiveBayesClassifier
from motion_detection import MotionClassifier


def test_feature_extraction():
    """Test DCT feature extraction."""
    print("Testing feature extraction...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Extract features
    features = extract_dct_features(test_image, block_size=5, num_coefficients=10)
    
    print(f"  [OK] Extracted {len(features)} feature vectors")
    print(f"  [OK] Feature shape: {features.shape}")
    
    # Test quantization
    quantized = quantize_features(features, num_bins=32)
    print(f"  [OK] Quantized features shape: {quantized.shape}")
    print(f"  [OK] Quantized range: [{quantized.min()}, {quantized.max()}]")
    
    assert features.shape[1] == 10, "Feature dimension mismatch"
    assert quantized.max() < 32, "Quantization out of range"
    
    print("  [OK] Feature extraction test passed!\n")


def test_naive_bayes():
    """Test Naive Bayes classifier."""
    print("Testing Naive Bayes classifier...")
    
    # Create synthetic training data
    num_samples = 100
    num_features = 10
    num_classes = 3
    
    # Generate features (simulate different classes)
    X_train = []
    y_train = []
    
    for c in range(num_classes):
        # Each class has different feature distributions
        features = np.random.randint(0, 32, (num_samples, num_features))
        # Add class-specific bias
        features[:, 0] = (features[:, 0] + c * 10) % 32
        X_train.append(features)
        y_train.extend([c] * num_samples)
    
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    
    # Train classifier
    classifier = NaiveBayesClassifier(num_classes, num_features, num_bins=32)
    classifier.fit(X_train, y_train)
    
    print(f"  [OK] Trained on {len(X_train)} samples")
    print(f"  [OK] Class priors: {classifier.class_priors}")
    
    # Test prediction
    X_test = np.random.randint(0, 32, (10, num_features))
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    
    print(f"  [OK] Predictions shape: {predictions.shape}")
    print(f"  [OK] Probabilities shape: {probabilities.shape}")
    print(f"  [OK] Sample predictions: {predictions[:5]}")
    
    assert len(predictions) == 10, "Prediction count mismatch"
    assert probabilities.shape == (10, num_classes), "Probability shape mismatch"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    
    print("  [OK] Naive Bayes test passed!\n")


def test_motion_detection():
    """Test motion detection."""
    print("Testing motion detection...")
    
    # Create two test frames with motion
    frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Shift frame2 to simulate translation
    frame2 = np.roll(frame1, 5, axis=1)
    
    classifier = MotionClassifier()
    
    # Compute optical flow
    flow = classifier.compute_optical_flow(frame1, frame2)
    
    print(f"  [OK] Optical flow shape: {flow.shape}")
    print(f"  [OK] Flow magnitude range: [{np.abs(flow).min():.2f}, {np.abs(flow).max():.2f}]")
    
    # Classify motion
    motion_map = classifier.classify_motion_type(flow, block_size=32)
    
    print(f"  [OK] Motion map shape: {motion_map.shape}")
    print(f"  [OK] Motion types detected: {np.unique(motion_map)}")
    
    # Test visualization
    colored_frame = classifier.visualize_motion(frame2, motion_map, block_size=32)
    
    print(f"  [OK] Visualization shape: {colored_frame.shape}")
    
    assert flow.shape[:2] == frame1.shape[:2], "Flow size mismatch"
    assert motion_map.shape[0] > 0 and motion_map.shape[1] > 0, "Motion map empty"
    
    print("  [OK] Motion detection test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    print()
    
    try:
        test_feature_extraction()
        test_naive_bayes()
        test_motion_detection()
        
        print("=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

