# Data Directory

This directory contains dataset information and preprocessing scripts.

## Guidelines

- **Do NOT store raw image/video files** in this repository
- Store only:
  - Dataset descriptions
  - Preprocessing scripts
  - Metadata files (CSV with video paths and labels)
  - Download instructions

## Dataset Structure

### Recommended Organization

```
data/
├── videos/                    # Raw videos (NOT in git)
│   ├── translation/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── rotation/
│   ├── zoom/
│   └── combined/
├── metadata/                  # Metadata files (IN git)
│   ├── train_labels.csv
│   ├── val_labels.csv
│   ├── test_labels.csv
│   └── dataset_info.txt
├── preprocess_videos.py       # Video preprocessing utilities
├── split_dataset.py          # Dataset splitting script
└── README.md                  # This file
```

## Scripts

### 1. `prepare_videos.py` - Video Preparation

Prepare and organize your video dataset:

```bash
# Resize videos
python prepare_videos.py --mode resize --input video.mp4 --output resized.mp4 --size 640x480

# Extract frames
python prepare_videos.py --mode extract-frames --input video.mp4 --output frames/ --interval 10

# Get video information
python prepare_videos.py --mode info --input video.mp4

# Organize videos by category
python prepare_videos.py --mode organize --input raw_videos/ --output organized/
```

### 2. `split_dataset.py` - Dataset Splitting

Split your dataset into train/validation/test sets to avoid data leakage:

```bash
python split_dataset.py \
    --input-dir data/videos \
    --output-dir data/metadata \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

This will:
- Split videos ensuring no video appears in multiple sets
- Maintain class distribution (stratified split)
- Create CSV files with video paths and labels
- Print statistics about the split

## Data Preparation Workflow

### Step 1: Collect Videos
- Gather videos for each category/motion type
- Ensure diversity (different lighting, backgrounds, angles)
- Minimum: 10-15 videos per category

### Step 2: Organize Videos
- Create category directories (translation, rotation, zoom, combined)
- Place videos in appropriate directories
- Use descriptive filenames

### Step 3: Preprocess (if needed)
- Cut long videos into 3-10 second segments
- Resize to consistent resolution (e.g., 640x480)
- Ensure consistent frame rate

### Step 4: Split Dataset
- Run `split_dataset.py` to create train/val/test splits
- Verify no data leakage (different videos in each set)
- Check label distribution is balanced

### Step 5: Document
- Fill in `dataset_info_template.txt` with your dataset information
- Save as `dataset_info.txt` in metadata directory

## Avoiding Data Leakage

**Critical Rules:**
1. **Never split the same video** across train/val/test
2. **Use different sources** when possible (different recording sessions, cameras, etc.)
3. **Maintain temporal separation** - if videos are from same session, keep entire session in one set
4. **Verify splits** - check that no video path appears in multiple CSV files

## Example Metadata CSV Format

The split script creates CSV files like:

```csv
video_path,label
data/videos/translation/video1.mp4,translation
data/videos/translation/video2.mp4,translation
data/videos/rotation/video1.mp4,rotation
...
```

## Loading Data in Code

```python
import pandas as pd
from code.video_processor import VideoProcessor

# Load metadata
train_df = pd.read_csv('data/metadata/train_labels.csv')

# Get video paths and labels
video_paths = train_df['video_path'].tolist()
labels = train_df['label'].tolist()

# Convert labels to numeric if needed
label_to_id = {label: idx for idx, label in enumerate(train_df['label'].unique())}
label_ids = [label_to_id[label] for label in labels]

# Train classifier
processor = VideoProcessor()
processor.train_style_classifier(video_paths, label_ids)
```

## Dataset Requirements

### For Style Classification
- **Minimum**: 2-3 categories
- **Per category**: 10-15 videos/images
- **Total**: 30-50 samples minimum

### For Motion Detection
- **Motion types**: Translation, Rotation, Zoom, Combined
- **Per type**: 3-5 videos minimum
- **Video length**: 3-10 seconds per clip

## Notes

- Keep raw videos outside the repository (use `.gitignore`)
- Only commit metadata and scripts
- Document data sources and collection methods
- Maintain reproducibility with random seeds
