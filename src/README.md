# Image Processing Project - Complete Pipeline Structure

## Overview

This project implements multiple pipelines for video activity recognition, based on Keren (2003) paper and extensions.

## Project Structure

```
src/
├── baseline/          # Baseline pipeline (2-class, non-overlapping blocks)
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── visualize_baseline.py
├── improved/          # Improved pipeline (2-class, 128x128 resolution, non-overlapping blocks)
│   ├── train_improved.py
│   ├── evaluate_improved.py
│   └── visualize_improved.py
├── multiclass/        # Multiclass pipeline (3-class, non-overlapping blocks)
│   ├── train_multiclass.py
│   ├── evaluate_multiclass.py
│   └── visualize_multiclass.py
├── deep_learning/     # Deep Learning pipeline (3D CNN for video action recognition)
│   ├── train_deep_learning.py
│   ├── evaluate_deep_learning.py
│   └── visualize_deep_learning.py
├── orchestration/     # Orchestration scripts
│   ├── run_pipeline_baseline.py
│   ├── run_pipeline_improved.py
│   ├── run_pipeline_multiclass.py
│   ├── run_all_pipelines.py
│   └── generate_final_comparison.py
└── utils/             # Shared utilities
    ├── block_extraction.py
    ├── dct_features.py
    ├── naive_bayes.py
    ├── metrics.py
    ├── video_loader.py
    └── visualization.py
```

## Quick Start

### Run Individual Pipeline

**Baseline:**
```bash
python src/orchestration/run_pipeline_baseline.py
```

**Improved:**
```bash
python src/orchestration/run_pipeline_improved.py
```

**Multiclass:**
```bash
python src/orchestration/run_pipeline_multiclass.py
```

### Run All Pipelines

```bash
python src/orchestration/run_all_pipelines.py
```

This will:
1. Prepare dataset (train/val/test split)
2. Run Baseline pipeline (train → evaluate → visualize)
3. Run Improved pipeline (train → evaluate → visualize)
4. Run Multiclass pipeline (train → evaluate → visualize)
5. Run Deep Learning pipeline (3D CNN)
6. Generate final comparison report

### Generate Final Comparison

After running all pipelines:

```bash
python src/orchestration/generate_final_comparison.py
```

This creates:
- `results/comparison/pipeline_comparison.png` - Comparison chart
- `results/comparison/comparison_metrics.csv` - Metrics table
- `results/comparison/all_pipelines_comparison.txt` - Detailed report

## Pipeline Details

### Baseline Pipeline
- **Classes**: WAVE_SIDE, WALKING (2 classes)
- **Blocks**: Non-overlapping (stride=5)
- **Methodology**: Exact Keren 2003 implementation
- **Results**: `results_baseline/` or `results/baseline/` (project root)

### Improved Pipeline
- **Classes**: WAVE_SIDE, WALKING (2 classes)
- **Resolution**: 128x128 (higher than baseline's 64x64)
- **Blocks**: Non-overlapping (stride=5)
- **Improvement**: Higher resolution for better spatial detail
- **Results**: `results_improved/` (project root)

### Multiclass Pipeline
- **Classes**: HELLO, WAVE_SIDE, WALKING (3 classes)
- **Resolution**: 128x128 (same as Improved)
- **Blocks**: Non-overlapping (stride=5, same as Improved)
- **Extension**: Same methodology as Improved, but with 3 classes (adds HELLO)
- **Results**: `results_multiclass/` (project root)

### Deep Learning Pipeline
- **Classes**: HELLO, WAVE_SIDE, WALKING (3 classes)
- **Architecture**: 3D CNN (R2Plus1D-18) for video action recognition
- **Results**: `results_deep_learning/` (project root)

## Output Files

Each pipeline generates:

### Training Outputs
- `classifier.pkl` - Trained model
- `training_config.pkl` - Training configuration
- `label_mapping.pkl` - Label mappings

### Evaluation Outputs
- `plots/accuracy_comparison.png` - Block/Frame/Video accuracy
- `plots/confusion_matrix.png` - Confusion matrix
- `plots/per_class_metrics.png` - Precision/Recall/F1 per class
- `plots/unclassified_blocks_pie.png` - Classified vs Unclassified
- `plots/confidence_distribution.png` - Confidence ratio histogram
- `plots/block_distribution_per_video.png` - Block distribution
- `per_video_breakdown.csv` - Per-video statistics
- `per_class_metrics.csv` - Per-class metrics
- `confusion_matrix_detailed.csv` - Confusion matrix
- `metrics_summary.txt` - Text summary
- `training_config.txt` - Training config

### Visualization Outputs
- `visualizations/*.mp4` - Colored block overlay videos (3 videos per class)

## Methodology

All pipelines use:
- **3D DCT** on 5×5×5 spatio-temporal blocks
- **MI-based feature selection** (top 10 features)
- **Threshold optimization** (maximize MI)
- **Binary feature binarization**
- **Bernoulli Naive Bayes** with P(C_i) = n_i/n priors
- **Activity filtering**: variance >= 20.0
- **Confidence filtering**: max_prob/min_prob >= 2.0

## Color Schemes

**Baseline/Improved (2-class):**
- YELLOW: hand_wave_side
- PURPLE: walking
- GRAY: unclassified

**Multiclass (3-class):**
- YELLOW: hand_wave_side
- PURPLE: walking
- BLUE: hand_wave_hello
- GRAY: unclassified

## References

- Keren, D. (2003). Recognizing image "style" and activities in video using local features and naive Bayes. *Pattern Recognition Letters*, 24(16), 2913-2922.
