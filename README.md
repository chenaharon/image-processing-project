# Image Processing Project - Video Action Recognition

## Overview

This project implements multiple pipelines for video activity recognition, based on Keren (2003) paper and extensions.

The project includes:
- **Baseline**: 2-class Naive Bayes classifier (reproduces Keren 2003)
- **Improved**: 2-class Naive Bayes with higher resolution (128x128)
- **Multiclass**: 3-class Naive Bayes (extends Improved with HELLO class)
- **Deep Learning**: 3D CNN (R2Plus1D-18) for modern video action recognition

## Project Structure

```
image-processing-project/
├── code/                    # Core implementation modules
│   ├── feature_extraction.py
│   ├── naive_bayes_classifier.py
│   ├── video_processor.py
│   └── generate_plots.py
├── src/                     # Pipeline implementations
│   ├── baseline/            # Baseline pipeline (2-class, 64x64)
│   ├── improved/            # Improved pipeline (2-class, 128x128)
│   ├── multiclass/          # Multiclass pipeline (3-class, 128x128)
│   ├── deep_learning/       # Deep Learning pipeline (3D CNN)
│   ├── orchestration/       # Pipeline orchestration scripts
│   └── utils/               # Shared utilities
├── data/                    # Dataset and metadata
│   ├── metadata/            # Train/val/test splits
│   ├── videos/              # Video files (not in repo)
│   └── unseen_videos/       # Directory for new test videos
├── results/                  # Results from all pipelines
│   ├── baseline/
│   ├── improved/
│   ├── multiclass/
│   ├── deep_learning/
│   └── comparison/          # Final comparison reports
├── docs/                     # Documentation
├── requirements.txt         # Python dependencies
└── run_pipeline.py          # Main user interface
```

## Quick Start

### Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate.ps1
   # Linux/Mac:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Pipelines

**Main Interface (Recommended):**
```bash
python run_pipeline.py
```

This provides an interactive menu to:
- Select a pipeline (Baseline, Improved, Multiclass, Deep Learning)
- Choose execution mode:
  - **Full Pipeline**: Training + Evaluation + Visualization
  - **Evaluation + Visualization**: Run on new unseen data

**Individual Pipelines:**
```bash
# Baseline
python src/orchestration/run_pipeline_baseline.py

# Improved
python src/orchestration/run_pipeline_improved.py

# Multiclass
python src/orchestration/run_pipeline_multiclass.py

# Deep Learning
python src/orchestration/run_pipeline_deep_learning.py
```

**Run All Pipelines:**
```bash
python src/orchestration/run_all_pipelines.py
```

**Generate Final Comparison:**
```bash
python src/orchestration/generate_final_comparison.py
```

## Pipeline Details

### Baseline Pipeline
- **Classes**: WAVE_SIDE, WALKING (2 classes)
- **Resolution**: 64x64
- **Blocks**: Non-overlapping (stride=5)
- **Methodology**: Exact Keren 2003 implementation
- **Results**: `results_baseline/`

### Improved Pipeline
- **Classes**: WAVE_SIDE, WALKING (2 classes)
- **Resolution**: 128x128 (higher than baseline)
- **Blocks**: Non-overlapping (stride=5)
- **Improvement**: Higher resolution for better spatial detail
- **Results**: `results_improved/`

### Multiclass Pipeline
- **Classes**: HELLO, WAVE_SIDE, WALKING (3 classes)
- **Resolution**: 128x128 (same as Improved)
- **Blocks**: Non-overlapping (stride=5, same as Improved)
- **Extension**: Same methodology as Improved, but with 3 classes
- **Results**: `results_multiclass/`

### Deep Learning Pipeline
- **Classes**: HELLO, WAVE_SIDE, WALKING (3 classes)
- **Architecture**: 3D CNN (R2Plus1D-18) for video action recognition
- **Results**: `results_deep_learning/`

## Output Files

Each pipeline generates:

### Training Outputs
- `classifier.pkl` or `model.pth` - Trained model
- `training_config.pkl` - Training configuration
- `label_mapping.pkl` - Label mappings

### Evaluation Outputs
- `plots/` - Various evaluation plots (accuracy, confusion matrix, etc.)
- `per_video_breakdown.csv` - Per-video statistics
- `per_class_metrics.csv` - Per-class metrics
- `confusion_matrix_detailed.csv` - Confusion matrix
- `metrics_summary.txt` - Text summary

### Visualization Outputs
- `visualizations/*.mp4` - Colored visualization videos

## Methodology

All Naive Bayes pipelines use:
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

**Deep Learning (3-class):**
- BLUE: hand_wave_hello
- YELLOW: hand_wave_side
- PURPLE: walking

## Testing on New Data

To evaluate and visualize on new unseen videos:

1. Place videos in `data/unseen_videos/` organized by class:
   - `data/unseen_videos/hand_wave_hello/`
   - `data/unseen_videos/hand_wave_side/`
   - `data/unseen_videos/walking/`

2. Run the main interface:
   ```bash
   python run_pipeline.py
   ```

3. Select a pipeline and choose "Evaluation + Visualization" mode

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+
- SciPy 1.7+
- pandas 1.3+
- PyTorch 1.9+ (for Deep Learning pipeline)
- torchvision (for Deep Learning pipeline)
- matplotlib, seaborn (for plotting)

See `requirements.txt` for complete list.

## Documentation

- **src/README.md**: Detailed pipeline structure
- **docs/BASELINE_METHODOLOGY.md**: Baseline implementation details
- **docs/BASELINE_PAPER_COMPARISON.md**: Comparison with Keren 2003 paper

## References

- Keren, D. (2003). Recognizing image "style" and activities in video using local features and naive Bayes. *Pattern Recognition Letters*, 24(16), 2913-2922.

## License

This project is for academic purposes only, following the course guidelines.
