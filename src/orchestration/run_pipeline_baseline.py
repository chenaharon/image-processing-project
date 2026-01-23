"""
Run Baseline Pipeline End-to-End
Executes: dataset preparation → training → evaluation → visualization
"""

import sys
from pathlib import Path
import subprocess
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data'))


def main():
    """Run complete baseline pipeline."""
    print("=" * 60)
    print("BASELINE PIPELINE - End-to-End Execution")
    print("=" * 60)
    
    # Step 0: Prepare dataset
    print("\n" + "=" * 60)
    print("STEP 0: Preparing Dataset")
    print("=" * 60)
    print("Checking video directories and updating train/val/test splits...")
    
    from prepare_dataset import prepare_dataset
    prepare_dataset(project_root)
    print("[OK] Dataset preparation completed")
    
    # Step 1: Train
    print("\n" + "=" * 60)
    print("STEP 1: Training Baseline Classifier")
    print("=" * 60)
    print("Starting training process...")
    
    # Prefer venv interpreter if available (ensures cv2, etc. are installed)
    venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    train_script = project_root / 'src' / 'baseline' / 'train_baseline.py'
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [python_exe, '-u', str(train_script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        print("  [ERROR] Training failed!")
        return 1
    print("  [OK] Training completed successfully")
    
    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating Baseline Classifier")
    print("=" * 60)
    print("  [2.1] Loading trained model...")
    print("  [2.2] Evaluating on validation set...")
    print("  [2.3] Evaluating on test set...")
    print("  [2.4] Generating plots and metrics...")
    print("  Starting evaluation process...")
    
    eval_script = project_root / 'src' / 'baseline' / 'evaluate_baseline.py'
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [python_exe, '-u', str(eval_script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        print("  [ERROR] Evaluation failed!")
        return 1
    print("  [OK] Evaluation completed successfully")
    
    # Step 3: Visualize (select top 3 videos from each class)
    print("\n" + "=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60)
    print("  [3.1] Selecting top videos from each class...")
    print("  [3.2] Generating colored block overlays...")
    print("  [3.3] Saving visualization videos...")
    print("  Starting visualization process...")
    
    # Check both possible paths
    results_dir = project_root / 'results_baseline'
    if not results_dir.exists():
        results_dir = project_root / 'results' / 'baseline'
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Load per-video breakdown to find best videos
    breakdown_path = results_dir / 'per_video_breakdown.csv'
    if breakdown_path.exists():
        breakdown_df = pd.read_csv(breakdown_path)
        
        # Load both val and test labels to find video paths
        val_df = pd.read_csv(project_root / 'data' / 'metadata' / 'val_labels.csv')
        test_df = pd.read_csv(project_root / 'data' / 'metadata' / 'test_labels.csv')
        all_labels_df = pd.concat([val_df, test_df], ignore_index=True)
        
        # Get top 3 videos from each class
        for label in ['walking', 'hand_wave_side']:
            label_videos = breakdown_df[breakdown_df['true_label'] == label].copy()
            if len(label_videos) > 0:
                # Sort by classified_percentage or accuracy
                if 'classified_percentage' in label_videos.columns:
                    label_videos = label_videos.sort_values('classified_percentage', ascending=False)
                elif 'correct' in label_videos.columns:
                    label_videos = label_videos.sort_values('correct', ascending=False)
                
                top_videos = label_videos.head(3)
                
                for idx, (_, row) in enumerate(top_videos.iterrows(), 1):
                    video_name = row.get('video_name', f'{label}_{idx}')
                    
                    # Find video path in both val and test sets
                    video_match = all_labels_df[all_labels_df['video_path'].str.contains(video_name, na=False, regex=False)]
                    
                    if len(video_match) > 0:
                        video_path = Path(video_match.iloc[0]['video_path'])
                        if not video_path.exists():
                            # Try to find relative to project root
                            video_path = project_root / video_match.iloc[0]['video_path']
                        
                        output_path = vis_dir / f"{video_name}_baseline.mp4"
                        
                        vis_script = project_root / 'src' / 'baseline' / 'visualize_baseline.py'
                        cmd = [
                            python_exe, str(vis_script),
                            '--video', str(video_path),
                            '--output', str(output_path),
                            '--classifier', str(results_dir / 'classifier.pkl'),
                            '--config', str(results_dir / 'training_config.pkl')
                        ]
                        
                        print(f"  Visualizing {label} video {idx}/3: {video_name}")
                        result = subprocess.run(cmd, cwd=str(project_root), check=False, 
                                               capture_output=True, text=True)
                        if result.returncode == 0:
                            print(f"    [OK] Saved: {output_path.name}")
                        else:
                            print(f"    [WARNING] Visualization may have issues for {video_name}")
                            if result.stderr:
                                print(f"      Error: {result.stderr[:200]}")
                    else:
                        print(f"    [WARNING] Could not find video path for {video_name}")
    else:
        print("  [WARNING] per_video_breakdown.csv not found. Skipping visualization.")
    
    print("\n" + "=" * 60)
    print("Baseline Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"\nResults saved in: {results_dir}")
    print(f"Visualizations saved in: {vis_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
