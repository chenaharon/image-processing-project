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
from sklearn.model_selection import train_test_split


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
    Uses stratified splitting to maintain class distribution.
    
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
    
    # Prepare data for splitting
    all_videos = []
    all_labels = []
    
    for category, videos in categories.items():
        for video in videos:
            all_videos.append(video)
            all_labels.append(category)
    
    print(f"\nTotal videos: {len(all_videos)}")
    print(f"Categories: {list(categories.keys())}")
    
    # Check if we have enough samples for stratified split
    # Need at least 2 samples per class for stratified split
    min_samples_per_class = min([len([l for l in all_labels if l == cat]) for cat in set(all_labels)])
    use_stratify = min_samples_per_class >= 2
    
    if not use_stratify:
        print(f"Warning: Some classes have < 2 samples. Using non-stratified split.")
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_videos, all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels if use_stratify else None,
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    
    # Check if we can stratify the second split
    min_samples_in_temp = min([len([l for l in y_temp if l == cat]) for cat in set(y_temp)])
    use_stratify_temp = min_samples_in_temp >= 2
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        stratify=y_temp if use_stratify_temp else None,
        random_state=random_seed
    )
    
    # Print statistics
    print(f"\nSplit results:")
    print(f"  Training:   {len(X_train)} videos ({len(X_train)/len(all_videos)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} videos ({len(X_val)/len(all_videos)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} videos ({len(X_test)/len(all_videos)*100:.1f}%)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
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
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
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

