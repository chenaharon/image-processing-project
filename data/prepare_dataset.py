"""
Central Dataset Preparation Script
Performs all dataset checks and splitting for all pipelines.

This script:
1. Checks if video directories have been updated
2. Verifies videos exist in directories
3. Splits dataset into TRAIN/VAL/TEST with fixed seed
4. Returns paths to appropriate results directories
"""

import os
import random
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fixed seed for reproducibility (as per prompt: seed=42, 70/15/15 split)
DATASET_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Fixed: 70% train, 15% val, 15% test


def check_video_directories_updated(video_dir: Path, metadata_dir: Path) -> bool:
    """
    Check if video directories have been updated since last split.
    
    Args:
        video_dir: Directory containing video categories
        metadata_dir: Directory containing metadata files
    
    Returns:
        True if videos were updated, False otherwise
    """
    if not video_dir.exists():
        return True  # Need to create if doesn't exist
    
    # Check if metadata files exist
    train_path = metadata_dir / "train_labels.csv"
    val_path = metadata_dir / "val_labels.csv"
    test_path = metadata_dir / "test_labels.csv"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        return True  # Need to create if don't exist
    
    # Get most recent modification time of metadata files
    metadata_time = max(
        train_path.stat().st_mtime,
        val_path.stat().st_mtime,
        test_path.stat().st_mtime
    )
    
    # Check if any video file is newer than metadata
    video_dir = Path(video_dir)
    for category_dir in video_dir.iterdir():
        if category_dir.is_dir():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                for video_file in category_dir.glob(ext):
                    if video_file.stat().st_mtime > metadata_time:
                        return True
    
    return False


def verify_videos_exist(video_dir: Path) -> Tuple[bool, Dict[str, int]]:
    """
    Verify that videos exist in directories.
    
    Args:
        video_dir: Directory containing video categories
    
    Returns:
        Tuple of (all_exist, category_counts)
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return False, {}
    
    category_counts = {}
    all_exist = True
    
    for category_dir in video_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            videos = []
            
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos.extend(category_dir.glob(ext))
            
            category_counts[category_name] = len(videos)
            
            if len(videos) == 0:
                print(f"Warning: No videos found in category: {category_name}")
                all_exist = False
    
    return all_exist, category_counts


def get_videos_by_category(video_dir: Path) -> Dict[str, list]:
    """
    Collect all videos organized by category/motion type.
    
    Args:
        video_dir: Root directory containing category subdirectories
    
    Returns:
        Dictionary mapping category to list of video paths
    """
    video_dir = Path(video_dir)
    categories = {}
    
    # Look for subdirectories (categories)
    for category_dir in video_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            videos = []
            
            # Find all video files
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos.extend(category_dir.glob(ext))
            
            if videos:
                categories[category_name] = [str(v) for v in sorted(videos)]
                print(f"Found {len(videos)} videos in category: {category_name}")
    
    return categories


def split_videos(video_dir: Path,
                train_ratio: float = TRAIN_RATIO,
                val_ratio: float = VAL_RATIO,
                test_ratio: float = TEST_RATIO,
                random_seed: int = DATASET_SEED) -> Dict[str, Tuple[list, list]]:
    """
    Split videos into train/val/test sets with fixed seed.
    
    Args:
        video_dir: Directory containing video categories
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility (fixed)
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (video_paths, labels)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Get videos by category
    categories = get_videos_by_category(video_dir)
    
    if not categories:
        raise ValueError(f"No videos found in {video_dir}")
    
    print(f"\nTotal videos: {sum(len(videos) for videos in categories.values())}")
    print(f"Categories: {list(categories.keys())}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Split each category separately to ensure exact distribution
    train_videos = []
    train_labels = []
    val_videos = []
    val_labels = []
    test_videos = []
    test_labels = []
    
    for category, videos in categories.items():
        # Shuffle videos for this category
        videos_shuffled = videos.copy()
        random.shuffle(videos_shuffled)
        
        total = len(videos_shuffled)
        # Calculate counts to match ratios: 70% train, 15% val, 15% test
        # Default behaviour: use floor for train, round for val, remainder to test.
        n_train = int(total * train_ratio)
        n_val = round(total * val_ratio)
        n_test = total - n_train - n_val

        # Special case: user requirement for 28 videos per class → 20/4/4
        if total == 28:
            n_train, n_val, n_test = 20, 4, 4
        
        # Ensure all are non-negative
        if n_test < 0:
            excess = -n_test
            if n_train > n_val:
                n_train -= excess
            else:
                n_val = max(1, n_val - excess)
            n_test = total - n_train - n_val
        
        # Ensure we have at least 1 in each set if possible
        if total >= 3:
            if n_train == 0:
                n_train = 1
                n_test = total - n_train - n_val
            if n_val == 0:
                n_val = 1
                n_test = total - n_train - n_val
            if n_test == 0:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)
        elif total == 2:
            n_train = 1
            n_val = 1
            n_test = 0
        elif total == 1:
            n_train = 1
            n_val = 0
            n_test = 0
        
        # Split videos
        train_vids = videos_shuffled[:n_train]
        val_vids = videos_shuffled[n_train:n_train + n_val]
        test_vids = videos_shuffled[n_train + n_val:]
        
        train_videos.extend(train_vids)
        train_labels.extend([category] * len(train_vids))
        val_videos.extend(val_vids)
        val_labels.extend([category] * len(val_vids))
        test_videos.extend(test_vids)
        test_labels.extend([category] * len(test_vids))
        
        print(f"\n{category}: {total} videos")
        print(f"  Train: {len(train_vids)} ({len(train_vids)/total*100:.1f}%)")
        print(f"  Val:   {len(val_vids)} ({len(val_vids)/total*100:.1f}%)")
        print(f"  Test:  {len(test_vids)} ({len(test_vids)/total*100:.1f}%)")
    
    # Print overall statistics
    total_all = len(train_videos) + len(val_videos) + len(test_videos)
    print(f"\nOverall split results:")
    print(f"  Training:   {len(train_videos)} videos ({len(train_videos)/total_all*100:.1f}%)")
    print(f"  Validation: {len(val_videos)} videos ({len(val_videos)/total_all*100:.1f}%)")
    print(f"  Test:       {len(test_videos)} videos ({len(test_videos)/total_all*100:.1f}%)")
    
    return {
        'train': (train_videos, train_labels),
        'val': (val_videos, val_labels),
        'test': (test_videos, test_labels)
    }


def save_split_metadata(splits: Dict[str, Tuple[list, list]],
                       output_dir: Path):
    """
    Save split metadata to CSV files.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Directory to save metadata files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, (video_paths, labels) in splits.items():
        df = pd.DataFrame({
            'video_path': video_paths,
            'label': labels
        })
        
        output_path = output_dir / f"{split_name}_labels.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {split_name} metadata to {output_path}")
        
        # Print label distribution
        print(f"  Label distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"    {label}: {count}")


def prepare_dataset(project_root: Optional[Path] = None,
                   video_dir: Optional[Path] = None,
                   metadata_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Main function to prepare dataset for all pipelines.
    
    This function:
    1. Checks if video directories have been updated
    2. Verifies videos exist
    3. Splits dataset into TRAIN/VAL/TEST with fixed seed
    4. Returns paths to results directories
    
    Args:
        project_root: Root directory of project (default: current directory)
        video_dir: Directory containing videos (default: project_root/data/videos)
        metadata_dir: Directory for metadata (default: project_root/data/metadata)
    
    Returns:
        Dictionary with 'results_dir', 'visualizations_dir' paths for each pipeline type
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    if video_dir is None:
        video_dir = project_root / "data" / "videos"
    
    if metadata_dir is None:
        metadata_dir = project_root / "data" / "metadata"
    
    project_root = Path(project_root)
    video_dir = Path(video_dir)
    metadata_dir = Path(metadata_dir)
    
    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Verify videos exist
    print("\nStep 1: Verifying videos exist...")
    all_exist, category_counts = verify_videos_exist(video_dir)
    if not all_exist:
        print("Warning: Some categories have no videos!")
    if not category_counts:
        raise ValueError(f"No videos found in {video_dir}")
    
    # Step 2: Always (re)split dataset at pipeline start
    print("\nStep 2: Splitting dataset into TRAIN/VAL/TEST (70/15/15)...")
    print(f"Using fixed seed = {DATASET_SEED} for reproducibility")
    splits = split_videos(
        video_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=DATASET_SEED
    )
    save_split_metadata(splits, metadata_dir)
    print("\n[OK] Dataset split completed and metadata CSVs updated")
    
    # Step 3: Return results directory paths
    results_paths = {
        'baseline': {
            'results_dir': project_root / "results_baseline",
            'visualizations_dir': project_root / "results" / "visualizations"
        },
        'improved': {
            'results_dir': project_root / "results_improved",
            'visualizations_dir': project_root / "results_improved" / "visualizations"
        },
        'multiclass': {
            'results_dir': project_root / "results_multiclass",
            'visualizations_dir': project_root / "results_multiclass" / "visualizations"
        }
    }
    
    # Create visualization directories
    for pipeline_type, paths in results_paths.items():
        paths['visualizations_dir'].mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)
    
    return results_paths


if __name__ == '__main__':
    # Can be run standalone
    prepare_dataset()
