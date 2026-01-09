# Image Processing Project - Style and Motion Recognition

## Overview

This project reproduces and extends the research from:
**"Recognizing image 'style' and activities in video using local features and naive Bayes"**
by Daniel Keren (Pattern Recognition Letters, 2003).

The system classifies image/video "style" and detects different types of motion in video sequences (translation, rotation, zoom, etc.), highlighting movement regions with color-coded visualization.

## Project Structure

```
image-processing-project/
├── code/              # Scripts and notebooks
│   ├── feature_extraction.py      # DCT-based feature extraction
│   ├── naive_bayes_classifier.py  # Naive Bayes classifier implementation
│   ├── motion_detection.py         # Motion type detection and classification
│   ├── video_processor.py         # Video loading and processing
│   └── run_experiment.py          # Main experiment script
├── data/              # Dataset info and preprocessing scripts
├── results/           # Generated images, plots, metrics
├── docs/              # Reports, slides, AI log
│   └── ai_assistance_log.md
├── requirements.txt   # Python dependencies
└── README.md          # This file
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

### Training the Classifier

Train the classifier on your dataset:

```bash
python train_classifier.py
```

This will:
- Load training videos from `data/metadata/train_labels.csv`
- Extract features using 5x5x5 spatio-temporal neighborhoods (as in paper)
- Train Naive Bayes classifier
- Save classifier to `results/classifier.pkl`

### Evaluating the Classifier

Evaluate the trained classifier on validation and test sets:

```bash
python evaluate_classifier.py
```

### Predicting Activity Type

Predict activity type for a new video:

```bash
python predict_video.py --video path/to/video.mp4
```

### Motion Detection and Visualization

Process a video to detect and visualize motion types:

```bash
python run_experiment.py --video path/to/video.mp4 --output results/output.mp4
```

### Command-line Options

- `--video`: Path to input video file (required)
- `--output`: Path to output video file (optional, defaults to `motion_<input_name>`)
- `--max-frames`: Maximum number of frames to process (optional)
- `--mode`: Processing mode - `motion`, `style`, or `both` (default: `motion`)

### Python API

```python
from code.video_processor import VideoProcessor

# Initialize processor
processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)

# Process video for motion detection
frames, visualized = processor.process_video_motion(
    'path/to/video.mp4',
    output_path='results/output.mp4'
)

# Train style classifier (requires labeled training data)
video_paths = ['video1.mp4', 'video2.mp4', ...]
labels = [0, 1, ...]  # Class labels
processor.train_style_classifier(video_paths, labels)
```

## Algorithm Overview

### 1. Feature Extraction (Following Keren 2003)
- **DCT-based features**: Extract Discrete Cosine Transform coefficients from 5x5 spatial blocks
- **5x5x5 spatio-temporal neighborhoods**: Extract features from 5x5 spatial blocks over 5 consecutive frames
- **64x64 resolution**: Frames are resized to 64x64 as in the paper
- **Quantization**: Convert continuous features to discrete bins for Naive Bayes

### 2. Style Classification
- **Naive Bayes Classifier**: Classifies image/video blocks based on local DCT features
- Assumes feature independence (naive assumption)
- Uses Laplace smoothing for robust probability estimation

### 3. Motion Detection
- **Optical Flow**: Computes dense optical flow between consecutive frames
- **Motion Classification**: Analyzes flow patterns to identify:
  - **Translation**: Uniform directional motion
  - **Rotation**: Circular motion patterns
  - **Zoom**: Radial motion from/to center
  - **Combined**: Mixed or complex motions
- **Visualization**: Color-codes motion regions:
  - Black: Static
  - Blue: Translation
  - Green: Rotation
  - Red: Zoom
  - Cyan: Combined

## Reproducing Results

### Phase 1-3: Setup and Data Preparation
1. Review the paper and related literature
2. Set up GitHub repository (this repo)
3. Prepare or collect datasets as described in the paper

### Phase 4: Algorithm Implementation
Run the basic algorithm:
```bash
python run_experiment.py --video data/videos/category/video.mp4
```

### Phase 5: Improvements
The system includes several enhancements beyond the paper:
- **Motion type classification**: Enhanced motion classification using optical flow analysis (translation, rotation, zoom, combined)
- **Color-coded motion visualization**: Visual highlighting of different motion types
- **Extended categories**: Classification of 3+ activity types (vs. 2 in paper: walking vs. hand waving)

### Phase 6-7: Visualization and Demo
The system automatically generates:
- Motion-classified video output
- Color-coded motion regions
- Reproducible results

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+
- SciPy 1.7+
- scikit-learn 1.0+
- matplotlib 3.4+

See `requirements.txt` for complete list.

## Data

- Store only scripts and metadata in `data/` directory
- Do NOT commit raw image/video files to repository
- Document dataset sources and preprocessing steps

## Results

Output files are saved to `results/`:
- Processed videos with motion visualization
- Plots and metrics
- Classification results

## Documentation

- `/docs/ai_assistance_log.md`: Log of all AI-assisted work (required by course guidelines)
- `/docs/METHODOLOGY.md`: Methodology verification against the paper
- `/docs/DATA_RECOMMENDATIONS.md`: Data collection recommendations for improved performance
- `/docs/reading_notes.pdf`: Reading notes and literature review (to be added)
- `/docs/final_report.pdf`: Final project report (to be added)
- `/docs/slides.pdf`: Presentation slides (to be added)

## Evaluation Criteria

- Understanding of paper & literature review (10 points)
- Data preparation & documentation (10 points)
- Algorithm reproduction (25 points)
- Original improvement (20 points)
- Motion classification & visualization (10 points)
- Working demo & reproducibility (10 points)
- Final detailed report (10 points)
- Presentation (5 points)


## License

This project is for academic purposes only, following the course guidelines.

## References

- Keren, D. (2003). Recognizing image "style" and activities in video using local features and naive Bayes. *Pattern Recognition Letters*, 24(16), 2913-2922.
- Paper URL: https://www.cs.haifa.ac.il/~dkeren/mypapers/style.pdf
