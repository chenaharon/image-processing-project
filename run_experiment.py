"""
Main experiment script - can be run from project root
Run this script to reproduce the paper's experiments and test the system.

Usage:
    python run_experiment.py --video <video_path> [--output <output_path>]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from video_processor import VideoProcessor
from motion_detection import MotionClassifier


def main():
    parser = argparse.ArgumentParser(description='Run image style and motion recognition experiment')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Path to output video file')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--mode', type=str, choices=['motion', 'style', 'both'], 
                       default='motion', help='Processing mode')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print("=" * 60)
    print("Image Style and Motion Recognition Experiment")
    print("Based on: Keren (2003) - Pattern Recognition Letters")
    print("=" * 60)
    
    # Initialize processor (using paper parameters: 5x5 blocks)
    processor = VideoProcessor(block_size=5, num_coefficients=10, num_bins=32)
    
    if args.mode in ['motion', 'both']:
        print("\n[1/2] Processing motion detection and classification...")
        frames, visualized_frames = processor.process_video_motion(
            args.video, 
            args.output if args.output else None,
            args.max_frames
        )
        
        if args.output:
            print(f"\nOutput video saved to: {args.output}")
        else:
            # Save default output
            output_path = str(Path(args.video).parent / f"motion_{Path(args.video).name}")
            processor.save_video(visualized_frames, output_path)
            print(f"\nOutput video saved to: {output_path}")
    
    if args.mode in ['style', 'both']:
        print("\n[2/2] Style classification requires training data.")
        print("To train the style classifier, use the VideoProcessor.train_style_classifier() method.")
        print("Example:")
        print("  processor = VideoProcessor()")
        print("  processor.train_style_classifier(video_paths, labels)")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

