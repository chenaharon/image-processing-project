"""
Complete Pipeline Script
Runs the full pipeline: dataset splitting → training → evaluation

This script executes the complete workflow:
1. Split dataset into train/val/test sets
2. Train the classifier
3. Evaluate on validation and test sets
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)
    print(f"Running: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[OK] {description} completed successfully")
    return result


def main():
    """Run the complete pipeline."""
    print("=" * 60)
    print("Complete Pipeline: Split -> Train -> Evaluate")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent
    python_exe = project_root / "venv" / "Scripts" / "python.exe"
    
    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        print("Please ensure virtual environment is set up correctly.")
        sys.exit(1)
    
    # Step 1: Split dataset
    video_dir = project_root / "data" / "videos"
    metadata_dir = project_root / "data" / "metadata"
    
    if not video_dir.exists():
        print(f"Error: Video directory not found at {video_dir}")
        sys.exit(1)
    
    print(f"\nVideo directory: {video_dir}")
    print(f"Metadata output: {metadata_dir}")
    
    split_cmd = f'"{python_exe}" "{project_root / "data" / "split_dataset.py"}" --input-dir "{video_dir}" --output-dir "{metadata_dir}" --train-ratio 0.75 --val-ratio 0.15 --test-ratio 0.1 --seed 42'
    
    run_command(split_cmd, "Step 1: Splitting Dataset")
    
    # Step 2: Train classifier
    train_cmd = f'"{python_exe}" "{project_root / "train_classifier.py"}"'
    
    run_command(train_cmd, "Step 2: Training Classifier")
    
    # Step 3: Evaluate classifier
    eval_cmd = f'"{python_exe}" "{project_root / "evaluate_classifier.py"}"'
    
    run_command(eval_cmd, "Step 3: Evaluating Classifier")
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Completed Successfully!")
    print("=" * 60)
    print("\nResults saved in:")
    print(f"  - Classifier: {project_root / 'results' / 'classifier.pkl'}")
    print(f"  - Training config: {project_root / 'results' / 'training_config.pkl'}")
    print(f"  - Metadata: {metadata_dir}")
    print("\nYou can now use predict_video.py to predict on new videos.")


if __name__ == '__main__':
    main()
