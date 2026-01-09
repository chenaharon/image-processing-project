"""
Example preprocessing script for datasets.
This is a template - modify according to your specific dataset needs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def preprocess_video(video_path: str, output_dir: str, 
                    target_size: Tuple[int, int] = None,
                    max_frames: int = None) -> List[str]:
    """
    Preprocess a video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save preprocessed frames
        target_size: Target frame size (width, height) or None to keep original
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of paths to preprocessed frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Resize if needed
        if target_size:
            frame = cv2.resize(frame, target_size)
        
        # Save frame
        frame_path = output_path / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        
        frame_count += 1
    
    cap.release()
    return frame_paths


def preprocess_image(image_path: str, output_path: str,
                    target_size: Tuple[int, int] = None) -> str:
    """
    Preprocess an image file.
    
    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image
        target_size: Target size (width, height) or None to keep original
    
    Returns:
        Path to preprocessed image
    """
    img = cv2.imread(image_path)
    
    if target_size:
        img = cv2.resize(img, target_size)
    
    cv2.imwrite(output_path, img)
    return output_path


if __name__ == '__main__':
    # Example usage
    print("Preprocessing example - modify according to your needs")
    # preprocess_video('input.mp4', 'data/processed/frames')

