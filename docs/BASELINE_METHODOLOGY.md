# Baseline Methodology - Paper Replication

This document describes the baseline implementation that replicates the methodology from the paper: "Recognizing image 'style' and activities in video using local features and naive Bayes" by Daniel Keren (2003).

## Overview

The baseline implements the paper's methodology for classifying video activities, specifically distinguishing between **walking** and **hand waving** (hand_wave_hello) activities, as described in Section 8 of the paper.

## Methodology

### 1. Data Preparation

#### Video Selection
- **Categories**: Only `hand_wave_hello` and `walking` categories are used (matching the paper's simplified scenario)
- **Balancing videos**: We balance the number of videos per class by taking the minimum number of videos available in any class
  - Example: If we have 11 `hand_wave_hello` videos and 11 `walking` videos, we use all 11 from each class
  - If we have 15 `hand_wave_hello` and 11 `walking`, we randomly sample 11 from each class

#### Video Length Normalization
- **Finding minimum length**: We check the frame count of all selected videos
- **Trimming**: All videos are trimmed to the same length (minimum frame count)
  - This ensures each video contributes the same number of features
  - Example: If the shortest video has 50 frames, all videos are trimmed to their first 50 frames
- **Rationale**: The paper does not specify how to handle videos of different lengths. By normalizing lengths, we ensure:
  - Equal contribution from each video
  - Balanced feature distribution (no need for post-extraction balancing)
  - Consistent temporal context across all videos

### 2. Feature Extraction

Following the paper's methodology (Section 4):

#### Spatio-Temporal Neighborhoods
- **5×5×5 neighborhoods**: Each feature is extracted from a 5×5 spatial block across 5 consecutive frames
- **Resolution**: All frames are resized to 64×64 pixels (as specified in Section 8 of the paper)
- **Spatial blocks**: 5×5 pixel blocks (non-overlapping, sliding window)
- **Temporal window**: 5 frames

#### DCT Feature Extraction
- **Method**: Extract 2D DCT from the center frame's 5×5 spatial block
- **Coefficients**: 10 DCT coefficients per block (low-frequency coefficients in zigzag pattern)
- **Process**:
  1. For each spatial position (i, j) in the 64×64 frame
  2. For each temporal window of 5 frames starting at time t
  3. Extract the 5×5×5 spatio-temporal volume
  4. Take the center frame (frame 2 of 5) and extract its 5×5 spatial block
  5. Apply 2D DCT to the 5×5 block
  6. Extract first 10 coefficients in zigzag pattern

#### Feature Quantization
- **Bins**: 32 quantization bins (paper does not specify, we use 32 as a standard choice)
- **Method**: 
  1. Compute global min/max for each feature dimension from training data
  2. Normalize features to [0, 1] range
  3. Quantize to 32 discrete bins (0-31)
- **Consistency**: Same min/max values are used for training, validation, and test sets

### 3. Training Process

#### Naive Bayes Classifier
Following the paper's methodology (Section 3):

**Class Priors**: P(C_i) = n_i / n
- `n_i` = number of features belonging to class i
- `n` = total number of features
- Since videos are balanced and same length, n_i = n_j for all i,j
- Therefore, P(C_i) = 0.5 for each class (balanced priors automatically)

**Conditional Probabilities**: P(feature_j = bin_k | class_i)
- Count how many features of class i have feature j = bin k
- Apply Laplace smoothing: (count + α) / (N_i + num_bins × α)
- α = 1.0 (standard Laplace smoothing)

**Training Process**:
1. Extract features from all training videos (with balanced lengths)
2. Quantize features using global min/max
3. Calculate class priors: P(C_i) = n_i / n
4. Calculate conditional probabilities: P(feature|class) for all features, classes, and bins
5. Save classifier, label mapping, and normalization parameters

**Note**: Naive Bayes is not an iterative algorithm - it directly computes probabilities from the data. There is no loss function or gradient descent.

### 4. Evaluation Process

#### Block-Level Prediction
- Each 5×5×5 neighborhood (block) gets a class prediction
- Block-level accuracy = percentage of correctly classified blocks

#### Video-Level Prediction
- For each video, use majority vote of all block predictions
- Video-level accuracy = percentage of correctly classified videos
- Confidence = average probability of the predicted class across all blocks

#### Pixel/Block Distribution (as in paper Section 8)
- Count how many blocks/pixels were classified as each class
- Report percentage distribution (e.g., "83% of classified pixels were labeled as 'walking'")
- This matches the paper's reporting style in Figures 6-7
- Note: This is distribution, not accuracy (paper does not report accuracy metrics)

#### Evaluation Sets
- **Validation set**: Used for model selection and hyperparameter tuning
- **Test set**: Final evaluation on unseen data
- Both sets use the same feature extraction and quantization as training (same min_frames, same min/max normalization)

## Implementation Details

### Hyperparameters (Matching Paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Spatial block size | 5×5 | Paper Section 4 |
| Temporal window | 5 frames | Paper Section 4 |
| Resolution | 64×64 | Paper Section 8 |
| DCT coefficients | 10 | Paper Section 4 (~10) |
| Quantization bins | 32 | Not specified in paper, standard choice |
| Laplace smoothing (α) | 1.0 | Standard practice |

### Key Design Decisions

1. **Video Length Normalization**: 
   - **Why**: Paper doesn't specify how to handle different video lengths
   - **How**: Trim all videos to minimum length
   - **Result**: Balanced feature distribution without post-processing

2. **No Feature Balancing**:
   - **Why**: Videos are already balanced (same number, same length)
   - **Result**: Each video contributes equally, no need for BALANCE_FEATURES

3. **Priors Calculation**:
   - **Why**: Follow paper's formula P(C_i) = n_i/n
   - **Result**: Automatically balanced (0.5/0.5) due to balanced videos

4. **Global Quantization**:
   - **Why**: Ensure consistent feature representation across train/val/test
   - **How**: Compute min/max from training set, apply to all sets

## Running the Baseline

### Training
```bash
python train_classifier.py
```

This will:
1. Load training videos
2. Balance number of videos per class
3. Find minimum video length
4. Extract features (all videos trimmed to same length)
5. Quantize features
6. Train Naive Bayes classifier
7. Save classifier and parameters

### Evaluation
```bash
python evaluate_classifier.py
```

This will:
1. Load trained classifier
2. Evaluate on validation set
3. Evaluate on test set
4. Report block-level and video-level accuracy

### Prediction
```bash
python predict_video.py --video path/to/video.mp4
```

This will:
1. Load trained classifier
2. Extract features from video (trimmed to training length)
3. Predict activity type
4. Display prediction and confidence

## Expected Results

Based on the paper (Section 8):
- The paper reports **pixel/block distribution** (not accuracy):
  - Figure 6 (walking): "Altogether, **83%** of the classified pixels were labeled as 'walking'"
  - Figure 7 (hand waving): "Altogether, **98%** of the classified pixels were labeled as 'hand waving'"
- **Note**: These percentages are distribution (how many pixels/blocks were classified as each class), NOT accuracy
- The paper does not report accuracy metrics (precision, recall, F1, etc.)
- Our implementation reports:
  - **Block-level accuracy**: Percentage of correctly classified blocks (for evaluation)
  - **Video-level accuracy**: Percentage of correctly classified videos (for evaluation)
  - **Pixel/Block Distribution**: Percentage of blocks classified as each class (matching paper's reporting style)

## Alignment with Paper

| Aspect | Paper | Our Implementation | Status |
|--------|-------|-------------------|--------|
| Categories | Walking vs hand waving | hand_wave_hello vs walking | ✅ Match |
| Spatio-temporal blocks | 5×5×5 | 5×5×5 | ✅ Match |
| Resolution | 64×64 | 64×64 | ✅ Match |
| DCT coefficients | ~10 | 10 | ✅ Match |
| Quantization | Discrete bins | 32 bins | ✅ Standard |
| Class priors | P(C_i) = n_i/n | P(C_i) = n_i/n | ✅ Match |
| Naive Bayes | Yes | Yes | ✅ Match |
| Video length handling | Not specified | Normalized to minimum | ✅ Reasonable |

## Notes

- The paper does not specify exact hyperparameters (number of DCT coefficients, quantization bins)
- We use standard choices (10 coefficients, 32 bins) that are reasonable for this task
- Video length normalization is our interpretation of how to handle different video lengths
- The baseline is designed to be reproducible and match the paper's methodology as closely as possible
