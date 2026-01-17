"""
Dataset Splitting Script
Splits videos into train/validation/test sets to avoid data leakage.
Ensures no video appears in multiple sets.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd


def get_videos_by_category(video_dir: str) -> Dict[str, List[str]]:
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
                categories[category_name] = [str(v) for v in videos]
                print(f"Found {len(videos)} videos in category: {category_name}")
    
    return categories


def split_videos(video_dir: str, 
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                random_seed: int = 42) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split videos into train/val/test sets to avoid data leakage.
    Uses exact splitting per category to ensure balanced distribution.
    
    Args:
        video_dir: Directory containing video categories
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    
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
        # Calculate counts to match ratios as closely as possible
        # Round each count, test gets remainder to ensure exact total
        n_train = round(total * train_ratio)
        n_val = round(total * val_ratio)
        n_test = total - n_train - n_val  # Remaining goes to test
        
        # Ensure all are non-negative
        if n_test < 0:
            # If test is negative, adjust train or val
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
                # Adjust train (prefer to keep train larger)
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


def save_split_metadata(splits: Dict[str, Tuple[List[str], List[str]]],
                       output_dir: str):
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


def create_directory_structure(base_dir: str, categories: List[str]):
    """
    Create directory structure for organized dataset.
    
    Args:
        base_dir: Base directory for dataset
        categories: List of category names
    """
    base_dir = Path(base_dir)
    
    for split in ['training', 'validation', 'test']:
        for category in categories:
            (base_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


def main():
    """Main function for dataset splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split video dataset into train/val/test')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing video categories')
    parser.add_argument('--output-dir', type=str, default='data/metadata',
                       help='Directory to save metadata files')
    parser.add_argument('--train-ratio', type=float, default=0.75,
                       help='Training set ratio (default: 0.75)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Splitting Tool")
    print("=" * 60)
    
    # Split videos
    splits = split_videos(
        args.input_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Save metadata
    save_split_metadata(splits, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset splitting completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

