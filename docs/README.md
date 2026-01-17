# Project Documentation

This directory contains documentation for the image processing project.

## Documents

- **BASELINE_METHODOLOGY.md**: Complete description of the baseline implementation that replicates the paper's methodology. Includes detailed explanation of training, evaluation, and how it aligns with the paper.

- **ai_assistance_log.md**: Log of all AI assistance used in the project, as required by the project guidelines.

## Project Structure

- `/code/`: Implementation code
- `/data/`: Dataset information and preprocessing scripts
- `/results/`: Trained models and evaluation results
- `/docs/`: Documentation (this directory)

## Quick Start

1. **Train the classifier**:
   ```bash
   python train_classifier.py
   ```

2. **Evaluate on test set**:
   ```bash
   python evaluate_classifier.py
   ```

3. **Predict on a new video**:
   ```bash
   python predict_video.py --video path/to/video.mp4
   ```

For detailed methodology, see [BASELINE_METHODOLOGY.md](BASELINE_METHODOLOGY.md).
