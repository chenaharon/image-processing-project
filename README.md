# Image Processing Project - Style and Motion Recognition

## Overview

This project reproduces the research from:
**"Recognizing image 'style' and activities in video using local features and naive Bayes"**
by Daniel Keren (Pattern Recognition Letters, 2003).

The baseline implementation classifies video activities, specifically distinguishing between **walking** and **hand waving** (hand_wave_hello) activities, as described in Section 8 of the paper.

## Project Structure

```
image-processing-project/
├── code/              # Implementation code
│   ├── feature_extraction.py      # DCT-based feature extraction
│   ├── naive_bayes_classifier.py  # Naive Bayes classifier
│   ├── motion_detection.py         # Motion type detection
│   └── video_processor.py         # Video loading and processing
├── data/              # Dataset info and preprocessing scripts
│   ├── metadata/      # Train/val/test splits
│   └── videos/       # Video files (not in repo)
├── results/           # Trained models and evaluation results
├── docs/              # Documentation
│   ├── BASELINE_METHODOLOGY.md  # Detailed methodology explanation
│   └── ai_assistance_log.md     # AI assistance log
├── requirements.txt   # Python dependencies
├── train_classifier.py    # Train the classifier
├── evaluate_classifier.py # Evaluate on test set
└── predict_video.py       # Predict on a new video
```

## Installation

### 1. Create and activate virtual environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Classifier (Baseline)

Train the baseline classifier matching the paper's methodology:

```bash
python train_classifier.py
```

This will:
1. Load training videos from `data/metadata/train_labels.csv`
2. Balance number of videos per class (take minimum)
3. Find minimum video length and trim all videos to same length
4. Extract features using 5×5×5 spatio-temporal neighborhoods
5. Quantize features to 32 discrete bins
6. Train Naive Bayes classifier with P(C_i) = n_i/n priors
7. Save classifier to `results/classifier.pkl`

### Evaluating the Classifier

Evaluate the trained classifier on validation and test sets:

```bash
python evaluate_classifier.py
```

This will:
- Load the trained classifier
- Evaluate on validation set (block-level and video-level accuracy)
- Evaluate on test set (block-level and video-level accuracy)
- Display results for each video

### Predicting Activity Type

Predict activity type for a new video:

```bash
python predict_video.py --video path/to/video.mp4
```

This will:
- Load the trained classifier
- Extract features from the video (trimmed to training length)
- Predict activity type (walking or hand_wave_hello)
- Display prediction and confidence

## Baseline Methodology

The baseline implementation follows the paper's methodology:

### Feature Extraction
- **5×5×5 spatio-temporal neighborhoods**: Extract features from 5×5 spatial blocks over 5 consecutive frames
- **64×64 resolution**: Frames resized to 64×64 (as in paper Section 8)
- **DCT coefficients**: 10 low-frequency DCT coefficients per block (zigzag pattern)
- **Quantization**: 32 discrete bins (0-31)

### Training
- **Naive Bayes Classifier**: P(C_i) = n_i/n (class priors based on feature distribution)
- **Balanced videos**: Equal number of videos per class, all trimmed to same length
- **Result**: Automatically balanced feature distribution (no post-processing needed)

### Evaluation
- **Block-level accuracy**: Percentage of correctly classified 5×5×5 neighborhoods
- **Video-level accuracy**: Percentage of correctly classified videos (majority vote)

For detailed methodology, see [docs/BASELINE_METHODOLOGY.md](docs/BASELINE_METHODOLOGY.md).

## Hyperparameters (Matching Paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Spatial block size | 5×5 | Paper Section 4 |
| Temporal window | 5 frames | Paper Section 4 |
| Resolution | 64×64 | Paper Section 8 |
| DCT coefficients | 10 | Paper Section 4 |
| Quantization bins | 32 | Standard choice |
| Laplace smoothing (α) | 1.0 | Standard practice |

## Data

- Videos are stored in `data/videos/` (not committed to repo)
- Metadata (train/val/test splits) in `data/metadata/`
- Only `hand_wave_hello` and `walking` categories are used (matching paper)

## Results

Output files are saved to `results/`:
- `classifier.pkl`: Trained Naive Bayes classifier
- `label_mapping.pkl`: Label to ID mapping
- `feature_normalization.pkl`: Min/max values for quantization
- `training_config.pkl`: Training configuration (video length, etc.)

## Documentation

- **docs/BASELINE_METHODOLOGY.md**: Complete description of baseline implementation
- **docs/ai_assistance_log.md**: Log of all AI-assisted work (required)

## Reproducing Results

1. **Prepare data**: Place videos in `data/videos/hand_wave_hello/` and `data/videos/walking/`
2. **Split dataset**: Run `python data/split_dataset.py` to create train/val/test splits
3. **Train**: Run `python train_classifier.py`
4. **Evaluate**: Run `python evaluate_classifier.py`

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+
- SciPy 1.7+
- pandas 1.3+
- tqdm 4.62+

See `requirements.txt` for complete list.

## References

- Keren, D. (2003). Recognizing image "style" and activities in video using local features and naive Bayes. *Pattern Recognition Letters*, 24(16), 2913-2922.
- Paper URL: https://www.cs.haifa.ac.il/~dkeren/mypapers/style.pdf

## License

This project is for academic purposes only, following the course guidelines.
