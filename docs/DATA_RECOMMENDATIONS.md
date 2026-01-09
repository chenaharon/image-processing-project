# Data Recommendations for Improved Performance

## Current Status

- **Total Videos**: ~15 videos
- **Categories**: 3 (hand_wave_hello, hand_wave_side, walking)
- **Videos per Category**: ~4-7
- **Current Performance**: 
  - Block-level accuracy: ~30%
  - Video-level accuracy: 0%

## Problem Analysis

The poor performance is primarily due to **insufficient training data**. With only 4-7 videos per category, the classifier cannot learn robust patterns.

## Recommended Dataset Size

### Minimum for Basic Performance
- **Per Category**: 15-20 videos
- **Total**: 45-60 videos minimum
- **Expected Performance**: 50-70% accuracy

### Recommended for Good Performance
- **Per Category**: 30-50 videos
- **Total**: 90-150 videos
- **Expected Performance**: 70-85% accuracy

### Ideal for Excellent Performance
- **Per Category**: 50+ videos
- **Total**: 150+ videos
- **Expected Performance**: 85-95% accuracy

## Data Collection Guidelines

### Video Specifications
- **Length**: 3-10 seconds per video
- **Resolution**: Any (will be resized to 64x64)
- **Frame Rate**: 15-30 fps
- **Format**: MP4, MOV, AVI

### Diversity Requirements
- **Lighting**: Different lighting conditions (indoor, outdoor, bright, dim)
- **Background**: Various backgrounds (plain, cluttered, different colors)
- **Camera Angle**: Different angles and distances
- **Subject**: If possible, different people performing the same action
- **Time of Day**: Different times if recording outdoors

### Quality Requirements
- Clear, focused videos
- Minimal camera shake
- Subject clearly visible
- Action fully contained in frame
- Consistent action execution

## Data Organization

```
data/videos/
├── hand_wave_hello/
│   ├── hello_001.mov
│   ├── hello_002.mov
│   └── ... (15-50 videos)
├── hand_wave_side/
│   ├── side_001.mov
│   └── ... (15-50 videos)
└── walking/
    ├── walk_001.mov
    └── ... (15-50 videos)
```

## Next Steps

1. **Collect More Data**: Aim for at least 15-20 videos per category
2. **Ensure Diversity**: Vary lighting, background, angles
3. **Re-split Dataset**: Run `split_dataset.py` with new data
4. **Re-train**: Run `train_classifier.py` with expanded dataset
5. **Re-evaluate**: Run `evaluate_classifier.py` to check improvement

## Expected Improvement

With 15-20 videos per category:
- Block-level accuracy: 50-70% (up from ~30%)
- Video-level accuracy: 40-60% (up from 0%)

With 30-50 videos per category:
- Block-level accuracy: 70-85%
- Video-level accuracy: 60-80%

