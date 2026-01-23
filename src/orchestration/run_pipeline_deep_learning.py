"""
Run Deep Learning Pipeline End-to-End
3D CNN (R2Plus1D-18) for modern video action recognition
"""

import sys
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data'))


def main():
    """Run complete Deep Learning pipeline."""
    print("=" * 60)
    print("DEEP LEARNING PIPELINE - End-to-End Execution")
    print("=" * 60)

    # Prefer venv interpreter if available
    venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
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
    print("STEP 1: Training Deep Learning Classifier")
    print("=" * 60)
    print("Starting training process (3D CNN)...")
    
    train_script = project_root / 'src' / 'deep_learning' / 'train_deep_learning.py'
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [python_exe, '-u', str(train_script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print("[ERROR] Training failed!")
        return 1
    print("[OK] Training completed successfully")
    
    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating Deep Learning Classifier")
    print("=" * 60)
    print("Starting evaluation...")
    
    eval_script = project_root / 'src' / 'deep_learning' / 'evaluate_deep_learning.py'
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        [python_exe, '-u', str(eval_script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print("[ERROR] Evaluation failed!")
        return 1
    print("[OK] Evaluation completed successfully")

    # Step 3: Visualize
    print("\n" + "=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60)
    print("Creating video visualizations with class predictions...")
    vis_script = project_root / 'src' / 'deep_learning' / 'visualize_deep_learning.py'
    process = subprocess.Popen(
        [python_exe, '-u', str(vis_script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output line by line
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print("[ERROR] Visualization failed!")
        return 1
    print("[OK] Visualization completed successfully")

    print("\nDeep Learning Pipeline Completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
