"""
Plot Generation Module
Generates all evaluation plots and reports for classifier evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def compute_per_class_metrics(y_true, y_pred, num_classes, id_to_label):
    """Compute precision, recall, F1-score per class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    metrics = []
    for i in range(num_classes):
        label_name = id_to_label.get(i, f"class_{i}")
        metrics.append({
            'class': label_name,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        })
    
    return pd.DataFrame(metrics)


def plot_accuracy_comparison(block_acc, frame_acc, video_acc, output_path):
    """Plot accuracy comparison: Block, Frame, Video level."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Block-level', 'Frame-level', 'Video-level']
    accuracies = [block_acc * 100, frame_acc * 100, video_acc * 100]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(metrics, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison: Block vs Frame vs Video Level', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved accuracy comparison plot to {output_path}")


def plot_confusion_matrix(y_true, y_pred, id_to_label, output_path):
    """Plot confusion matrix as heatmap."""
    num_classes = len(id_to_label)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    # Create labels
    labels = [id_to_label.get(i, f"Class {i}") for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix plot to {output_path}")
    
    # Also save as CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    csv_path = output_path.with_suffix('.csv')
    cm_df.to_csv(csv_path)
    print(f"[OK] Saved confusion matrix CSV to {csv_path}")


def plot_per_class_metrics(metrics_df, output_path):
    """Plot per-class metrics: Precision, Recall, F1-Score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    precision = metrics_df['precision'].values * 100
    recall = metrics_df['recall'].values * 100
    f1 = metrics_df['f1_score'].values * 100
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics: Precision, Recall, F1-Score', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['class'].values, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved per-class metrics plot to {output_path}")
    
    # Also save as CSV
    csv_path = output_path.with_suffix('.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"[OK] Saved per-class metrics CSV to {csv_path}")


def plot_unclassified_blocks_pie(classified, unclassified, output_path):
    """Plot pie chart: Classified vs Unclassified blocks."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [classified, unclassified]
    labels = ['Classified', 'Unclassified']
    colors = ['#2ecc71', '#95a5a6']
    explode = (0.05, 0)  # Slight explode for classified
    
    total = classified + unclassified
    percentages = [f'{classified/total*100:.1f}%', f'{unclassified/total*100:.1f}%']
    labels_with_pct = [f'{label}\n({pct}, {size:,} blocks)' 
                       for label, pct, size in zip(labels, percentages, sizes)]
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels_with_pct, 
                                       colors=colors, autopct='', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Block Classification Status', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved unclassified blocks pie chart to {output_path}")


def plot_confidence_distribution(confidence_ratios, threshold=2.0, output_path=None):
    """Plot histogram of confidence ratio distribution."""
    if len(confidence_ratios) == 0:
        print("Warning: No confidence ratios to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out extreme values for better visualization
    filtered_ratios = confidence_ratios[confidence_ratios <= 20]  # Cap at 20 for visualization
    
    ax.hist(filtered_ratios, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Confidence Ratio (max/min probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Ratio Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics text
    mean_ratio = np.mean(confidence_ratios)
    median_ratio = np.median(confidence_ratios)
    above_threshold = np.sum(confidence_ratios >= threshold)
    pct_above = above_threshold / len(confidence_ratios) * 100
    
    stats_text = f'Mean: {mean_ratio:.2f}\nMedian: {median_ratio:.2f}\nAbove threshold: {pct_above:.1f}%'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved confidence distribution plot to {output_path}")
    plt.close()


def plot_block_distribution_per_video(video_stats, id_to_label, output_path):
    """Plot stacked bar chart: block distribution per video."""
    if len(video_stats) == 0:
        print("Warning: No video stats to plot")
        return
    
    # Prepare data
    df = pd.DataFrame(video_stats)
    
    # Show all videos (no limit)
    
    num_videos = len(df)
    
    fig, ax = plt.subplots(figsize=(max(12, num_videos * 0.8), 8))
    
    video_names = df['video_name'].values
    video_labels = df['true_label'].values
    total_blocks = df['total_blocks'].values
    classified_blocks = df['classified_blocks'].values
    unclassified_blocks = df['unclassified_blocks'].values
    
    x = np.arange(len(video_names))
    width = 0.6
    
    colors_map = {
        'hand_wave_hello': '#3498db',
        'hand_wave_side': '#2ecc71',
        'hand_waving': '#2ecc71',
        'walking': '#e74c3c'
    }
    
    # Stack: classified blocks (colored by true label) + unclassified blocks (gray)
    for i, (name, label, total, classified, unclassified) in enumerate(
        zip(video_names, video_labels, total_blocks, classified_blocks, unclassified_blocks)
    ):
        color = colors_map.get(label, '#95a5a6')
        # Classified blocks (bottom)
        ax.bar(i, classified, width, color=color, alpha=0.8, label=label if i == 0 or label not in [l for l in video_labels[:i]] else '')
        # Unclassified blocks (top, stacked)
        ax.bar(i, unclassified, width, bottom=classified, color='#95a5a6', alpha=0.5)
    
    ax.set_xlabel('Video', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Blocks', fontsize=12, fontweight='bold')
    ax.set_title('Block Distribution per Video (Classified vs Unclassified)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([Path(name).stem for name in video_names], rotation=45, ha='right')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#95a5a6', alpha=0.5, label='Unclassified')]
    for label, color in colors_map.items():
        if label in video_labels:
            legend_elements.append(Patch(facecolor=color, alpha=0.8, label=label.replace('_', ' ').title()))
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved block distribution per video plot to {output_path}")


def generate_all_plots(results: Dict, label_to_id: Dict, output_dir: Path, 
                      confidence_ratios: Optional[np.ndarray] = None,
                      video_stats: Optional[List] = None):
    """
    Generate all evaluation plots and reports.
    
    Args:
        results: Dictionary with evaluation results
        label_to_id: Label to ID mapping
        output_dir: Directory to save plots
        confidence_ratios: Optional array of confidence ratios
        video_stats: Optional list of per-video statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)
    
    # Extract results
    block_acc = results.get('block_accuracy', 0)
    frame_acc = results.get('frame_accuracy', 0)
    video_acc = results.get('video_accuracy', 0)
    block_predictions = results.get('block_predictions', np.array([]))
    block_labels = results.get('block_labels', np.array([]))
    total_blocks = results.get('total_blocks', 0)
    classified_blocks = results.get('classified_blocks', 0)
    unclassified_blocks = results.get('unclassified_blocks', 0)
    
    print("\n" + "=" * 60)
    print("Generating Evaluation Plots and Reports")
    print("=" * 60)
    
    # 1. Accuracy comparison
    if block_acc > 0 or frame_acc > 0 or video_acc > 0:
        plot_accuracy_comparison(block_acc, frame_acc, video_acc, 
                                output_dir / 'accuracy_comparison.png')
        print(f"  [OK] accuracy_comparison.png")
    else:
        print(f"  [SKIP] accuracy_comparison.png (no accuracy data)")
    
    # 2. Confusion matrix
    if len(block_predictions) > 0 and len(block_labels) > 0:
        plot_confusion_matrix(block_labels, block_predictions, id_to_label,
                             output_dir / 'confusion_matrix.png')
        print(f"  [OK] confusion_matrix.png")
    else:
        print(f"  [SKIP] confusion_matrix.png (no predictions)")
    
    # 3. Per-class metrics
    if len(block_predictions) > 0 and len(block_labels) > 0:
        metrics_df = compute_per_class_metrics(block_labels, block_predictions, 
                                              num_classes, id_to_label)
        plot_per_class_metrics(metrics_df, output_dir / 'per_class_metrics.png')
        print(f"  [OK] per_class_metrics.png")
    else:
        print(f"  [SKIP] per_class_metrics.png (no predictions)")
    
    # 4. Unclassified blocks pie chart
    if total_blocks > 0:
        plot_unclassified_blocks_pie(classified_blocks, unclassified_blocks,
                                    output_dir / 'unclassified_blocks_pie.png')
        print(f"  [OK] unclassified_blocks_pie.png")
    else:
        print(f"  [SKIP] unclassified_blocks_pie.png (no blocks)")
    
    # 5. Confidence distribution
    if confidence_ratios is not None and len(confidence_ratios) > 0:
        plot_confidence_distribution(confidence_ratios, threshold=2.0,
                                   output_path=output_dir / 'confidence_distribution.png')
        print(f"  [OK] confidence_distribution.png")
    else:
        print(f"  [SKIP] confidence_distribution.png (no confidence ratios)")
    
    # 6. Block distribution per video
    if video_stats is not None and len(video_stats) > 0:
        plot_block_distribution_per_video(video_stats, id_to_label,
                                         output_dir / 'block_distribution_per_video.png')
        print(f"  [OK] block_distribution_per_video.png")
    else:
        print(f"  [SKIP] block_distribution_per_video.png (no video stats)")
    
    print("\n[OK] All plots generated successfully!")


def save_per_video_breakdown(video_predictions_list, label_to_id, output_path):
    """Save per-video breakdown to CSV."""
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    breakdown = []
    for item in video_predictions_list:
        if len(item) >= 8:
            video_name, predictions, true_label, num_high_conf, total_blocks, low_conf_blocks, frame_preds, frame_labels = item
        elif len(item) >= 6:
            video_name, predictions, true_label, num_high_conf, total_blocks, low_conf_blocks = item
            frame_preds, frame_labels = [], []
        elif len(item) >= 5:
            video_name, predictions, true_label, num_high_conf, total_blocks = item
            low_conf_blocks = total_blocks - num_high_conf
            frame_preds, frame_labels = [], []
        else:
            continue
        
        # Video-level prediction (majority vote)
        if len(predictions) > 0:
            video_pred = np.bincount(predictions).argmax()
            pred_label = id_to_label.get(video_pred, "unknown")
            confidence = num_high_conf / total_blocks if total_blocks > 0 else 0
        else:
            pred_label = "unknown"
            confidence = 0
        
        # Frame-level accuracy
        if len(frame_preds) > 0 and len(frame_labels) > 0:
            frame_acc = np.mean(np.array(frame_preds) == np.array(frame_labels))
        else:
            frame_acc = 0
        
        breakdown.append({
            'video_name': video_name,
            'true_label': true_label,
            'predicted_label': pred_label,
            'correct': true_label == pred_label,
            'confidence': confidence,
            'total_blocks': total_blocks,
            'classified_blocks': num_high_conf,
            'unclassified_blocks': low_conf_blocks,
            'classified_percentage': (num_high_conf / total_blocks * 100) if total_blocks > 0 else 0,
            'frame_accuracy': frame_acc * 100
        })
    
    df = pd.DataFrame(breakdown)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved per-video breakdown to {output_path}")


def save_training_config(config, output_path):
    """Save training configuration to text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Block Parameters:\n")
        f.write(f"  Block size: {config.get('block_size', 'N/A')}\n")
        f.write(f"  Stride: {config.get('stride', 'N/A')}\n")
        f.write(f"  Temporal window: {config.get('temporal_window', 'N/A')}\n")
        f.write(f"  Spatial resolution: {config.get('spatial_size', 'N/A')}\n\n")
        
        f.write("Feature Parameters:\n")
        f.write(f"  DCT coefficients: {config.get('num_coefficients', 'N/A')}\n")
        f.write(f"  Selected features: {config.get('num_features', 'N/A')}\n")
        f.write(f"  Quantization bins: {config.get('num_bins', 'N/A')}\n")
        f.write(f"  Use binary features: {config.get('use_binary_features', 'N/A')}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Number of classes: {config.get('num_classes', 'N/A')}\n")
        f.write(f"  Min frames: {config.get('min_frames', 'N/A')}\n")
        f.write(f"  Activity threshold: {config.get('min_activity', 'N/A')}\n\n")
        
        if config.get('selected_feature_indices') is not None:
            f.write("Selected Feature Indices:\n")
            f.write(f"  {config.get('selected_feature_indices')}\n\n")
        
        if config.get('optimal_thresholds') is not None:
            f.write("Optimal Thresholds:\n")
            for i, threshold in enumerate(config.get('optimal_thresholds')):
                f.write(f"  Feature {i}: {threshold:.4f}\n")
    
    print(f"[OK] Saved training config to {output_path}")


def save_metrics_summary(results, label_to_id, output_path):
    """Save detailed metrics summary to text file."""
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Evaluation Metrics Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall accuracy
        f.write("Overall Accuracy:\n")
        f.write(f"  Block-level: {results.get('block_accuracy', 0)*100:.2f}%\n")
        f.write(f"  Frame-level: {results.get('frame_accuracy', 0)*100:.2f}%\n")
        f.write(f"  Video-level: {results.get('video_accuracy', 0)*100:.2f}%\n\n")
        
        # Block statistics
        total_blocks = results.get('total_blocks', 0)
        classified_blocks = results.get('classified_blocks', 0)
        unclassified_blocks = results.get('unclassified_blocks', 0)
        
        f.write("Block Statistics:\n")
        f.write(f"  Total blocks: {total_blocks:,}\n")
        f.write(f"  Classified blocks: {classified_blocks:,} ({classified_blocks/total_blocks*100:.2f}%)\n")
        f.write(f"  Unclassified blocks: {unclassified_blocks:,} ({unclassified_blocks/total_blocks*100:.2f}%)\n\n")
        
        # Per-class metrics
        if len(results.get('block_predictions', [])) > 0:
            block_predictions = results.get('block_predictions')
            block_labels = results.get('block_labels')
            metrics_df = compute_per_class_metrics(block_labels, block_predictions, 
                                                  num_classes, id_to_label)
            
            f.write("Per-Class Metrics:\n")
            for _, row in metrics_df.iterrows():
                f.write(f"  {row['class']}:\n")
                f.write(f"    Precision: {row['precision']*100:.2f}%\n")
                f.write(f"    Recall: {row['recall']*100:.2f}%\n")
                f.write(f"    F1-Score: {row['f1_score']*100:.2f}%\n")
                f.write(f"    Support: {row['support']:,}\n\n")
    
    print(f"[OK] Saved metrics summary to {output_path}")
