"""
Run All Pipelines
Orchestrates execution of all 4 pipelines:
1. Baseline (2-class, non-overlapping)
2. Improved (2-class, overlapping + smoothing)
3. Multiclass (3-class, non-overlapping)
4. Deep Learning (3-class, 3D CNN)

Each pipeline:
- Prepares dataset
- Trains model
- Evaluates model
- Generates visualizations
"""

import sys
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import dataset preparation
sys.path.insert(0, str(project_root / 'data'))
from prepare_dataset import prepare_dataset


def run_pipeline(pipeline_name: str, script_name: str):
    """Run a single pipeline script."""
    print("\n" + "=" * 60)
    print(f"Running {pipeline_name} Pipeline")
    print("=" * 60)
    
    script_path = project_root / script_name
    
    if not script_path.exists():
        print(f"Warning: {script_name} not found, skipping...")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        print(f"\n[OK] {pipeline_name} pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {pipeline_name} pipeline failed with error: {e}")
        return False


def main():
    """Run all pipelines."""
    print("=" * 60)
    print("RUNNING ALL PIPELINES")
    print("=" * 60)
    
    # Step 0: Prepare dataset
    print("\n" + "=" * 60)
    print("Step 0: Preparing Dataset")
    print("=" * 60)
    
    prepare_dataset(project_root)
    
    # Step 1: Baseline (complete pipeline)
    print("\n" + "=" * 60)
    print("Step 1: Baseline Pipeline")
    print("=" * 60)
    baseline_script = project_root / 'src' / 'orchestration' / 'run_pipeline_baseline.py'
    subprocess.run([sys.executable, str(baseline_script)], cwd=str(project_root), check=False)
    
    # Step 2: Improved (complete pipeline)
    print("\n" + "=" * 60)
    print("Step 2: Improved Pipeline")
    print("=" * 60)
    improved_script = project_root / 'src' / 'orchestration' / 'run_pipeline_improved.py'
    subprocess.run([sys.executable, str(improved_script)], cwd=str(project_root), check=False)
    
    # Step 3: Multiclass (complete pipeline)
    print("\n" + "=" * 60)
    print("Step 3: Multiclass Pipeline")
    print("=" * 60)
    multiclass_script = project_root / 'src' / 'orchestration' / 'run_pipeline_multiclass.py'
    subprocess.run([sys.executable, str(multiclass_script)], cwd=str(project_root), check=False)
    
    # Step 4: Deep Learning (complete pipeline)
    print("\n" + "=" * 60)
    print("Step 4: Deep Learning Pipeline (3D CNN)")
    print("=" * 60)
    deep_learning_script = project_root / 'src' / 'orchestration' / 'run_pipeline_deep_learning.py'
    subprocess.run([sys.executable, str(deep_learning_script)], cwd=str(project_root), check=False)
    
    # Step 5: Generate final comparison
    print("\n" + "=" * 60)
    print("Step 5: Generating Final Comparison")
    print("=" * 60)
    
    comparison_script = project_root / "src" / "orchestration" / "generate_final_comparison.py"
    if comparison_script.exists():
        subprocess.run([sys.executable, str(comparison_script)], cwd=str(project_root))
    else:
        # Try old location
        comparison_script = project_root / "generate_final_comparison.py"
        if comparison_script.exists():
            subprocess.run([sys.executable, str(comparison_script)], cwd=str(project_root))
    
    print("\n" + "=" * 60)
    print("ALL PIPELINES COMPLETED")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
