"""
Generate Final Comparison
Compares all 4 pipelines and generates comprehensive comparison reports.
Creates:
- pipeline_comparison.png (grouped bar chart)
- comparison_metrics.csv
- all_pipelines_comparison.txt (detailed report)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_results(pipeline_name: str, results_dir: Path):
    """Load results from a pipeline."""
    results_dir = Path(results_dir)
    
    metrics = {
        'pipeline_name': pipeline_name,
        'results_dir': results_dir
    }
    
    # Try to load metrics summary
    metrics_file = results_dir / 'metrics_summary.txt'
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Block-level:' in line:
                    try:
                        metrics['block_accuracy'] = float(line.split(':')[1].strip().replace('%', '')) / 100
                    except:
                        pass
                elif 'Frame-level:' in line:
                    try:
                        metrics['frame_accuracy'] = float(line.split(':')[1].strip().replace('%', '')) / 100
                    except:
                        pass
                elif 'Video-level:' in line:
                    try:
                        metrics['video_accuracy'] = float(line.split(':')[1].strip().replace('%', '')) / 100
                    except:
                        pass
    
    # Try to load per-class metrics CSV
    per_class_file = results_dir / 'per_class_metrics.csv'
    if per_class_file.exists():
        metrics['per_class_metrics'] = pd.read_csv(per_class_file)
    
    # Try to load confusion matrix
    cm_file = results_dir / 'confusion_matrix_detailed.csv'
    if cm_file.exists():
        metrics['confusion_matrix'] = pd.read_csv(cm_file, index_col=0)
    
    return metrics


def plot_pipeline_comparison(all_metrics: dict, output_path: Path):
    """Plot comparison of all pipelines."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    pipelines = ['Baseline', 'Improved', 'Multiclass', 'DeepLearning']
    metrics_names = ['Block-level', 'Frame-level', 'Video-level']
    
    x = np.arange(len(metrics_names))
    width = 0.2
    
    # Extract accuracies for each pipeline
    pipeline_data = {}
    for pipeline in pipelines:
        metrics = all_metrics.get(pipeline, {})
        pipeline_data[pipeline] = [
            metrics.get('block_accuracy', 0) * 100,
            metrics.get('frame_accuracy', 0) * 100,
            metrics.get('video_accuracy', 0) * 100
        ]
    
    # Colors
    colors = {
        'Baseline': '#FF6B6B',
        'Improved': '#4ECDC4',
        'Multiclass': '#45B7D1',
        'DeepLearning': '#FFA07A'
    }
    
    # Plot bars
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    for idx, pipeline in enumerate(pipelines):
        bars = ax.bar(x + offsets[idx], pipeline_data[pipeline], width, 
                     label=pipeline, color=colors[pipeline], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Accuracy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: All Pipelines', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc='upper left')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved pipeline comparison plot to {output_path}")


def save_comparison_metrics(all_metrics: dict, output_path: Path):
    """Save comparison metrics to CSV."""
    rows = []
    
    pipelines = ['Baseline', 'Improved', 'Multiclass', 'DeepLearning']
    for pipeline in pipelines:
        metrics = all_metrics.get(pipeline, {})
        rows.append({
            'Metric': pipeline,
            'Block_Level_Accuracy (%)': metrics.get('block_accuracy', 0) * 100,
            'Frame_Level_Accuracy (%)': metrics.get('frame_accuracy', 0) * 100,
            'Video_Level_Accuracy (%)': metrics.get('video_accuracy', 0) * 100,
            'Classes': 2 if pipeline in ['Baseline', 'Improved'] else 3,
            'Block_Overlap': 'None' if pipeline in ['Baseline', 'Multiclass'] else 'Stride2' if pipeline == 'Improved' else 'N/A',
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved comparison metrics to {output_path}")


def save_comparison_report(all_metrics: dict, output_path: Path):
    """Save detailed comparison report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FINAL COMPARISON - ALL PIPELINES\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("[EXECUTIVE SUMMARY]\n")
        f.write("  Goal: Reproduce Keren 2003 paper and extend with improvements + Deep Learning\n\n")
        
        # Ranking
        pipelines = ['Baseline', 'Improved', 'Multiclass', 'DeepLearning']
        video_accs = [(p, all_metrics.get(p, {}).get('video_accuracy', 0)) for p in pipelines]
        video_accs.sort(key=lambda x: x[1], reverse=True)
        
        f.write("Ranking by accuracy (video-level):\n")
        for rank, (pipeline, acc) in enumerate(video_accs, 1):
            star = " ⭐ (BEST)" if rank == 1 else ""
            f.write(f"  {rank}. {pipeline}: {acc*100:.1f}%{star}\n")
        f.write("\n")
        
        # Detailed comparison for each pipeline
        for pipeline in pipelines:
            metrics = all_metrics.get(pipeline, {})
            f.write(f"[PIPELINE: {pipeline.upper()}]\n")
            
            if pipeline == 'Baseline':
                f.write("  Classes: 2 (WAVE_SIDE, WALKING)\n")
                f.write("  Block strategy: Non-overlapping (stride=5)\n")
            elif pipeline == 'Improved':
                f.write("  Classes: 2 (WAVE_SIDE, WALKING)\n")
                f.write("  Resolution: 128x128 (higher than baseline)\n")
                f.write("  Block strategy: Non-overlapping (stride=5)\n")
            elif pipeline == 'Multiclass':
                f.write("  Classes: 3 (HELLO, WAVE_SIDE, WALKING)\n")
                f.write("  Resolution: 128x128 (same as Improved)\n")
                f.write("  Block strategy: Non-overlapping (stride=5, same as Improved)\n")
                f.write("  Note: Same methodology as Improved, but with 3 classes\n")
            elif pipeline == 'DeepLearning':
                f.write("  Classes: 3 (HELLO, WAVE_SIDE, WALKING)\n")
                f.write("  Architecture: 3D CNN (R2Plus1D-18) for video action recognition\n")
            
            f.write("\n  Results:\n")
            f.write(f"    Block-level:  {metrics.get('block_accuracy', 0)*100:.1f}%\n")
            f.write(f"    Frame-level:  {metrics.get('frame_accuracy', 0)*100:.1f}%\n")
            f.write(f"    Video-level:  {metrics.get('video_accuracy', 0)*100:.1f}%\n")
            f.write("\n")
        
        # Comparison table
        f.write("[COMPARISON TABLE]\n")
        f.write("Pipeline         | Block   | Frame   | Video   | Classes\n")
        f.write("-" * 60 + "\n")
        for pipeline in pipelines:
            metrics = all_metrics.get(pipeline, {})
            classes = 2 if pipeline in ['Baseline', 'Improved'] else 3
            f.write(f"{pipeline:15} | {metrics.get('block_accuracy', 0)*100:6.1f}% | "
                   f"{metrics.get('frame_accuracy', 0)*100:6.1f}% | "
                   f"{metrics.get('video_accuracy', 0)*100:6.1f}% | {classes}\n")
        f.write("\n")
        
        # Recommendations
        f.write("[RECOMMENDATIONS]\n\n")
        f.write("For this project:\n")
        f.write("  1. **Submit Baseline** as primary result (faithfully reproduces Keren 2003)\n")
        f.write("  2. **Improved as main improvement** (overlapping blocks + smoothing)\n")
        f.write("  3. **Multiclass as extension** (demonstrates 3-class capability)\n")
        f.write("  4. **Deep Learning as modern approach** (3D CNN for video action recognition)\n\n")
        
        f.write("For production deployment:\n")
        best_pipeline = video_accs[0][0]
        best_acc = video_accs[0][1] * 100
        f.write(f"  → Use {best_pipeline} ({best_acc:.1f}% accuracy)\n")
        f.write("  → If interpretability required → use Improved Naive Bayes\n")
    
    print(f"[OK] Saved comparison report to {output_path}")


def main():
    """Generate final comparison."""
    print("=" * 60)
    print("Generating Final Pipeline Comparison")
    print("=" * 60)
    
    # Load results from all pipelines
    all_metrics = {}
    
    pipelines = {
        'Baseline': project_root / 'results' / 'baseline',
        'Improved': project_root / 'results_improved',
        'Multiclass': project_root / 'results_multiclass',
        'DeepLearning': project_root / 'results_deep_learning'
    }
    
    # Also check old paths for backward compatibility
    if not pipelines['Baseline'].exists():
        pipelines['Baseline'] = project_root / 'results_baseline'
    if not pipelines['Improved'].exists():
        pipelines['Improved'] = project_root / 'results_improved'
    if not pipelines['Multiclass'].exists():
        pipelines['Multiclass'] = project_root / 'results_multiclass'
    
    for pipeline_name, results_dir in pipelines.items():
        if results_dir.exists():
            metrics = load_results(pipeline_name, results_dir)
            if metrics:
                all_metrics[pipeline_name] = metrics
                print(f"Loaded results for {pipeline_name}")
        else:
            print(f"Warning: Results directory not found for {pipeline_name}: {results_dir}")
    
    if len(all_metrics) == 0:
        print("Error: No pipeline results found!")
        return 1
    
    # Create output directory
    output_dir = project_root / 'results' / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plot
    plot_pipeline_comparison(all_metrics, output_dir / 'pipeline_comparison.png')
    
    # Save comparison metrics
    save_comparison_metrics(all_metrics, output_dir / 'comparison_metrics.csv')
    
    # Save comparison report
    save_comparison_report(all_metrics, output_dir / 'all_pipelines_comparison.txt')
    
    print("\n" + "=" * 60)
    print("Comparison generation completed!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
