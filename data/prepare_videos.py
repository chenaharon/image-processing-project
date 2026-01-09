"""
Video Preparation Script
Helps prepare and organize video datasets for the project.
Includes functions for resizing videos, extracting frames, getting video info, and organizing by category.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm


def resize_video(input_video: str,
                output_video: str,
                target_size: Tuple[int, int] = (640, 480)) -> str:
    """
    Resize video to target dimensions.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        target_size: Target (width, height)
    
    Returns:
        Path to output video
    """
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, target_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, target_size)
        out.write(resized)
    
    cap.release()
    out.release()
    
    return output_video


def extract_frames(video_path: str,
                   output_dir: str,
                   frame_interval: int = 1) -> List[str]:
    """
    Extract frames from video at specified interval.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
    
    Returns:
        List of paths to extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def organize_videos_by_category(source_dir: str,
                               target_dir: str,
                               category_mapping: dict):
    """
    Organize videos into category directories.
    
    Args:
        source_dir: Source directory with videos
        target_dir: Target directory for organized structure
        category_mapping: Dictionary mapping video patterns to categories
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    videos = list(source_dir.glob('*.mp4')) + \
             list(source_dir.glob('*.avi')) + \
             list(source_dir.glob('*.mov'))
    
    for video in videos:
        # Determine category based on filename patterns
        category = None
        for pattern, cat in category_mapping.items():
            if pattern.lower() in video.name.lower():
                category = cat
                break
        
        if category is None:
            category = 'other'
        
        # Create category directory and move/copy video
        category_dir = target_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = category_dir / video.name
        # Copy instead of move to preserve original
        import shutil
        shutil.copy2(video, target_path)
        print(f"Organized {video.name} -> {category}/{video.name}")


def main():
    """Main function for video preparation."""
    parser = argparse.ArgumentParser(description='Prepare and organize video datasets')
    parser.add_argument('--mode', type=str, 
                       choices=['resize', 'extract-frames', 'info', 'organize'],
                       required=True, help='Operation mode')
    parser.add_argument('--input', type=str, required=True,
                       help='Input video or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory or file')
    parser.add_argument('--size', type=str, default='640x480',
                       help='Target size WxH (for resize mode)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Frame interval (for extract-frames mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'resize':
        width, height = map(int, args.size.split('x'))
        resize_video(args.input, args.output, (width, height))
    
    elif args.mode == 'extract-frames':
        extract_frames(args.input, args.output, frame_interval=args.interval)
    
    elif args.mode == 'info':
        info = get_video_info(args.input)
        if info:
            print(f"Video: {args.input}")
            print(f"  Resolution: {info['width']}x{info['height']}")
            print(f"  FPS: {info['fps']:.2f}")
            print(f"  Frames: {info['frame_count']}")
            print(f"  Duration: {info['duration']:.2f}s")
    
    elif args.mode == 'organize':
        # Example category mapping - modify as needed
        category_mapping = {
            'translation': 'translation',
            'rotation': 'rotation',
            'zoom': 'zoom',
            'combined': 'combined'
        }
        organize_videos_by_category(args.input, args.output, category_mapping)


if __name__ == '__main__':
    main()

