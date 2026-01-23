"""
Main User Interface for Running Pipelines
==========================================

This script provides a user-friendly interface to run any pipeline:
- Full pipeline (training + evaluation + visualization)
- Evaluation + Visualization on new unseen data

Available Pipelines:
1. Baseline - 2-class Naive Bayes with non-overlapping blocks
2. Improved - 2-class Naive Bayes with overlapping blocks
3. Multiclass - 3-class Naive Bayes with non-overlapping blocks
4. Deep Learning - 3D CNN for video action recognition
"""

import sys
import os
from pathlib import Path
import subprocess
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data'))

# Prefer venv interpreter if available
venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'
python_exe = str(venv_python) if venv_python.exists() else sys.executable


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_menu():
    """Print the main menu."""
    print_header("PIPELINE SELECTION")
    print("\nAvailable Pipelines:")
    print("  1. Baseline (2-class, Naive Bayes, 64x64, non-overlapping blocks)")
    print("  2. Improved (2-class, Naive Bayes, 128x128, non-overlapping blocks)")
    print("  3. Multiclass (3-class, Naive Bayes, 128x128, same as Improved + HELLO class)")
    print("  4. Deep Learning (3-class, 3D CNN)")
    print("\n  0. Exit")
    print("\n" + "-" * 70)


def print_mode_menu():
    """Print the mode selection menu."""
    print_header("EXECUTION MODE")
    print("\nSelect execution mode:")
    print("  1. Full Pipeline (Training + Evaluation + Visualization)")
    print("  2. Evaluation + Visualization (on new unseen data)")
    print("\n  0. Back to pipeline selection")
    print("\n" + "-" * 70)


def get_pipeline_info(choice):
    """Get pipeline information based on user choice."""
    pipelines = {
        '1': {
            'name': 'Baseline',
            'description': '2-class, 64x64, non-overlapping blocks',
            'script': 'src/orchestration/run_pipeline_baseline.py',
            'eval_script': 'src/baseline/evaluate_baseline.py',
            'vis_script': 'src/baseline/visualize_baseline.py',
            'results_dir': 'results_baseline',
            'model_file': 'classifier.pkl',
            'config_file': 'training_config.pkl'
        },
        '2': {
            'name': 'Improved',
            'description': '2-class, 128x128, non-overlapping blocks',
            'script': 'src/orchestration/run_pipeline_improved.py',
            'eval_script': 'src/improved/evaluate_improved.py',
            'vis_script': 'src/improved/visualize_improved.py',
            'results_dir': 'results_improved',
            'model_file': 'classifier.pkl',
            'config_file': 'training_config.pkl'
        },
        '3': {
            'name': 'Multiclass',
            'description': '3-class, 128x128, same as Improved + HELLO class',
            'script': 'src/orchestration/run_pipeline_multiclass.py',
            'eval_script': 'src/multiclass/evaluate_multiclass.py',
            'vis_script': 'src/multiclass/visualize_multiclass.py',
            'results_dir': 'results_multiclass',
            'model_file': 'classifier.pkl',
            'config_file': 'training_config.pkl'
        },
        '4': {
            'name': 'Deep Learning',
            'description': '3-class, 3D CNN (R2Plus1D-18)',
            'script': 'src/orchestration/run_pipeline_deep_learning.py',
            'eval_script': 'src/deep_learning/evaluate_deep_learning.py',
            'vis_script': 'src/deep_learning/visualize_deep_learning.py',
            'results_dir': 'results_deep_learning',
            'model_file': 'model.pth',
            'config_file': 'training_config.pkl'
        }
    }
    return pipelines.get(choice)


def run_full_pipeline(pipeline_info):
    """Run the complete pipeline (training + evaluation + visualization)."""
    print_header(f"RUNNING FULL {pipeline_info['name'].upper()} PIPELINE")
    
    script_path = project_root / pipeline_info['script']
    if not script_path.exists():
        print(f"[ERROR] Pipeline script not found: {script_path}")
        return False
    
    print(f"Executing: {script_path}")
    print("\nThis will run:")
    print("  - Dataset preparation")
    print("  - Model training")
    print("  - Model evaluation")
    print("  - Visualization generation")
    print("\nStarting pipeline...\n")
    
    try:
        process = subprocess.Popen(
            [python_exe, '-u', str(script_path)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print_header(f"{pipeline_info['name'].upper()} PIPELINE COMPLETED SUCCESSFULLY")
            return True
        else:
            print(f"\n[ERROR] Pipeline failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Failed to run pipeline: {e}")
        return False


def check_model_exists(pipeline_info):
    """Check if trained model exists."""
    results_dir = project_root / pipeline_info['results_dir']
    model_path = results_dir / pipeline_info['model_file']
    config_path = results_dir / pipeline_info['config_file']
    
    if not model_path.exists() or not config_path.exists():
        return False, results_dir
    return True, results_dir


def get_unseen_data_dir():
    """Get or create directory for unseen data."""
    unseen_dir = project_root / 'data' / 'unseen_videos'
    unseen_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in ['hand_wave_hello', 'hand_wave_side', 'walking']:
        class_dir = unseen_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    return unseen_dir


def prepare_unseen_data_metadata(unseen_dir):
    """Create metadata CSV for unseen videos."""
    import pandas as pd
    import cv2
    
    metadata = []
    
    for class_name in ['hand_wave_hello', 'hand_wave_side', 'walking']:
        class_dir = unseen_dir / class_name
        for video_file in class_dir.glob('*.mp4'):
            if not video_file.is_file():
                continue
            
            # Get video info
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                metadata.append({
                    'video_path': str(video_file),
                    'label': class_name,
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height
                })
    
    if not metadata:
        return None
    
    df = pd.DataFrame(metadata)
    metadata_file = unseen_dir / 'unseen_labels.csv'
    df.to_csv(metadata_file, index=False)
    return metadata_file


def run_evaluation_and_visualization(pipeline_info):
    """Run evaluation and visualization on unseen data."""
    print_header(f"EVALUATION + VISUALIZATION - {pipeline_info['name'].upper()}")
    
    # Check if model exists
    model_exists, results_dir = check_model_exists(pipeline_info)
    if not model_exists:
        print(f"\n[ERROR] Trained model not found!")
        print(f"Expected location: {results_dir / pipeline_info['model_file']}")
        print(f"\nPlease run the full pipeline first to train the model.")
        return False
    
    print(f"[OK] Found trained model at: {results_dir}")
    
    # Check for unseen data
    unseen_dir = get_unseen_data_dir()
    metadata_file = prepare_unseen_data_metadata(unseen_dir)
    
    if not metadata_file or not metadata_file.exists():
        print(f"\n[INFO] No unseen videos found in: {unseen_dir}")
        print(f"\nTo use this mode:")
        print(f"  1. Place video files in: {unseen_dir}")
        print(f"  2. Organize videos by class in subdirectories:")
        print(f"     - {unseen_dir / 'hand_wave_hello'}")
        print(f"     - {unseen_dir / 'hand_wave_side'}")
        print(f"     - {unseen_dir / 'walking'}")
        print(f"  3. Run this script again")
        return False
    
    print(f"[OK] Found unseen videos. Metadata: {metadata_file}")
    
    # Run evaluation
    print_header("STEP 1: EVALUATION")
    eval_script = project_root / pipeline_info['eval_script']
    
    if not eval_script.exists():
        print(f"[ERROR] Evaluation script not found: {eval_script}")
        return False
    
    print(f"Running evaluation on unseen data...")
    
    try:
        # For evaluation, we need to modify the script to use unseen data
        # This is a simplified version - in practice, you might need to pass arguments
        process = subprocess.Popen(
            [python_exe, '-u', str(eval_script)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**dict(os.environ), 'UNSEEN_DATA': str(metadata_file)}
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[WARNING] Evaluation may have issues, but continuing...")
    except Exception as e:
        print(f"\n[WARNING] Evaluation error: {e}, but continuing to visualization...")
    
    # Run visualization
    print_header("STEP 2: VISUALIZATION")
    vis_script = project_root / pipeline_info['vis_script']
    
    if not vis_script.exists():
        print(f"[ERROR] Visualization script not found: {vis_script}")
        return False
    
    # Load unseen data to visualize all videos
    import pandas as pd
    unseen_df = pd.read_csv(metadata_file)
    # Filter to supported classes based on pipeline
    if pipeline_info['name'] == 'Baseline':
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side'])].copy()
    elif pipeline_info['name'] == 'Improved':
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side'])].copy()
    elif pipeline_info['name'] == 'Multiclass':
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side', 'hand_wave_hello'])].copy()
    elif pipeline_info['name'] == 'Deep Learning':
        unseen_df = unseen_df[unseen_df['label'].isin(['walking', 'hand_wave_side', 'hand_wave_hello'])].copy()
    
    if len(unseen_df) == 0:
        print(f"[WARNING] No videos found with supported labels for {pipeline_info['name']} pipeline")
        return False
    
    print(f"Generating visualizations for {len(unseen_df)} unseen videos...")
    
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for idx, (_, row) in enumerate(unseen_df.iterrows(), 1):
        video_path = Path(row['video_path'])
        if not video_path.exists():
            # Try relative to project root
            video_path = project_root / row['video_path']
        
        if not video_path.exists():
            print(f"  [{idx}/{len(unseen_df)}] SKIP: {row.get('video_name', video_path.name)} - file not found")
            failed_count += 1
            continue
        
        video_name = video_path.stem
        output_path = vis_dir / f"{video_name}_baseline.mp4"
        
        # Build command
        cmd = [
            python_exe, '-u', str(vis_script),
            '--video', str(video_path),
            '--output', str(output_path),
            '--classifier', str(results_dir / pipeline_info['model_file']),
            '--config', str(results_dir / pipeline_info['config_file'])
        ]
        
        print(f"  [{idx}/{len(unseen_df)}] Visualizing: {video_name}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per video
            )
            
            if result.returncode == 0:
                print(f"    [OK] Saved: {output_path.name}")
                success_count += 1
            else:
                print(f"    [ERROR] Failed to visualize {video_name}")
                if result.stderr:
                    print(f"      Error: {result.stderr}")
                elif result.stdout:
                    # Sometimes errors go to stdout
                    error_lines = result.stdout.split('\n')
                    error_msg = '\n'.join([line for line in error_lines if 'Error' in line or 'Traceback' in line or 'Exception' in line][:10])
                    if error_msg:
                        print(f"      Error: {error_msg}")
                failed_count += 1
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] Timeout while visualizing {video_name}")
            failed_count += 1
        except Exception as e:
            print(f"    [ERROR] Exception: {e}")
            failed_count += 1
    
    print(f"\nVisualization Summary:")
    print(f"  Success: {success_count}/{len(unseen_df)}")
    print(f"  Failed: {failed_count}/{len(unseen_df)}")
    
    if success_count > 0:
        print_header(f"EVALUATION + VISUALIZATION COMPLETED")
        print(f"\nResults saved in: {results_dir}")
        print(f"Visualizations saved in: {vis_dir}")
        return True
    else:
        print(f"\n[ERROR] All visualizations failed")
        return False


def main():
    """Main entry point."""
    import os
    
    print_header("VIDEO ACTION RECOGNITION - PIPELINE RUNNER")
    print("\nWelcome! This tool allows you to:")
    print("  - Run complete pipelines (training + evaluation + visualization)")
    print("  - Evaluate and visualize on new unseen data")
    
    while True:
        print_menu()
        pipeline_choice = input("Select pipeline (1-4, 0 to exit): ").strip()
        
        if pipeline_choice == '0':
            print("\nExiting... Goodbye!")
            return 0
        
        pipeline_info = get_pipeline_info(pipeline_choice)
        if not pipeline_info:
            print("\n[ERROR] Invalid choice. Please select 1-4 or 0 to exit.")
            continue
        
        print_mode_menu()
        mode_choice = input("Select mode (1-2, 0 to go back): ").strip()
        
        if mode_choice == '0':
            continue
        elif mode_choice == '1':
            success = run_full_pipeline(pipeline_info)
            if success:
                input("\nPress Enter to continue...")
        elif mode_choice == '2':
            success = run_evaluation_and_visualization(pipeline_info)
            if success:
                input("\nPress Enter to continue...")
        else:
            print("\n[ERROR] Invalid choice. Please select 1-2 or 0 to go back.")
            continue


if __name__ == '__main__':
    sys.exit(main())
