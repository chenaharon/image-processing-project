"""
Metrics and Evaluation Utilities
Computes accuracy at block, frame, and video levels.
Generates confusion matrices and per-class metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_block_level_accuracy(predictions: List[int], true_labels: List[int],
                                 confidence_ratios: Optional[List[float]] = None,
                                 confidence_threshold: float = 2.0) -> Tuple[float, int, int]:
    """
    Compute block-level accuracy.
    
    Only counts blocks that passed confidence filtering (if provided).
    
    Args:
        predictions: Predicted class labels
        true_labels: True class labels
        confidence_ratios: Optional confidence ratios for each prediction
        confidence_threshold: Confidence threshold (default: 2.0)
    
    Returns:
        Tuple of (accuracy, num_correct, num_classified)
    """
    if confidence_ratios is not None:
        # Only count blocks that passed confidence filtering
        classified_mask = np.array(confidence_ratios) >= confidence_threshold
        predictions = np.array(predictions)[classified_mask]
        true_labels = np.array(true_labels)[classified_mask]
    else:
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
    
    if len(predictions) == 0:
        return 0.0, 0, 0
    
    num_correct = np.sum(predictions == true_labels)
    num_classified = len(predictions)
    accuracy = num_correct / num_classified if num_classified > 0 else 0.0
    
    return accuracy, num_correct, num_classified


def compute_frame_level_accuracy(block_predictions: List[List[int]],
                                 block_true_labels: List[List[int]],
                                 block_confidence_ratios: Optional[List[List[float]]] = None,
                                 confidence_threshold: float = 2.0) -> Tuple[float, int, int]:
    """
    Compute frame-level accuracy using majority vote.
    
    For each frame, take majority vote of classified blocks.
    
    Args:
        block_predictions: List of predictions per frame (each frame has list of block predictions)
        block_true_labels: List of true labels per frame
        block_confidence_ratios: Optional confidence ratios per frame
        confidence_threshold: Confidence threshold
    
    Returns:
        Tuple of (accuracy, num_correct_frames, num_total_frames)
    """
    frame_predictions = []
    frame_true_labels = []
    
    for frame_idx in range(len(block_predictions)):
        frame_preds = block_predictions[frame_idx]
        frame_labels = block_true_labels[frame_idx]
        
        if block_confidence_ratios is not None:
            frame_conf = block_confidence_ratios[frame_idx]
            # Filter by confidence
            classified_mask = np.array(frame_conf) >= confidence_threshold
            frame_preds = np.array(frame_preds)[classified_mask]
            frame_labels = np.array(frame_labels)[classified_mask]
        
        if len(frame_preds) == 0:
            continue  # Skip frames with no classified blocks
        
        # Majority vote
        unique, counts = np.unique(frame_preds, return_counts=True)
        frame_pred = unique[np.argmax(counts)]
        
        # True label is the same for all blocks in a frame (from same video)
        frame_true = frame_labels[0] if len(frame_labels) > 0 else -1
        
        if frame_true >= 0:
            frame_predictions.append(frame_pred)
            frame_true_labels.append(frame_true)
    
    if len(frame_predictions) == 0:
        return 0.0, 0, 0
    
    num_correct = np.sum(np.array(frame_predictions) == np.array(frame_true_labels))
    num_frames = len(frame_predictions)
    accuracy = num_correct / num_frames if num_frames > 0 else 0.0
    
    return accuracy, num_correct, num_frames


def compute_video_level_accuracy(block_predictions: List[int],
                                video_true_label: int,
                                block_confidence_ratios: Optional[List[float]] = None,
                                confidence_threshold: float = 2.0) -> Tuple[int, float]:
    """
    Compute video-level prediction using majority vote.
    
    Args:
        block_predictions: All block predictions for a video
        video_true_label: True label for the video
        block_confidence_ratios: Optional confidence ratios
        confidence_threshold: Confidence threshold
    
    Returns:
        Tuple of (predicted_label, confidence)
    """
    if block_confidence_ratios is not None:
        classified_mask = np.array(block_confidence_ratios) >= confidence_threshold
        block_predictions = np.array(block_predictions)[classified_mask]
    else:
        block_predictions = np.array(block_predictions)
    
    if len(block_predictions) == 0:
        return -1, 0.0
    
    # Majority vote
    unique, counts = np.unique(block_predictions, return_counts=True)
    predicted_label = unique[np.argmax(counts)]
    
    # Confidence = proportion of blocks voting for predicted class
    confidence = np.sum(block_predictions == predicted_label) / len(block_predictions)
    
    return predicted_label, confidence


def compute_per_class_metrics(y_true: List[int], y_pred: List[int],
                             class_names: List[str]) -> pd.DataFrame:
    """
    Compute precision, recall, F1-score per class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        DataFrame with columns: Class, Precision, Recall, F1-Score, Support
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Compute macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Build DataFrame
    data = []
    for i, class_name in enumerate(class_names):
        data.append({
            'Class': class_name,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-Score': f1[i],
            'Support': support[i]
        })
    
    # Add averages
    data.append({
        'Class': 'macro avg',
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1-Score': macro_f1,
        'Support': sum(support)
    })
    data.append({
        'Class': 'weighted avg',
        'Precision': weighted_precision,
        'Recall': weighted_recall,
        'F1-Score': weighted_f1,
        'Support': sum(support)
    })
    
    return pd.DataFrame(data)


def create_confusion_matrix(y_true: List[int], y_pred: List[int],
                           class_names: List[str]) -> np.ndarray:
    """
    Create confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    return confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
