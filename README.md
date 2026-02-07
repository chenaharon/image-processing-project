# Image Processing Project — Video Action Recognition

End-to-end README: run the project from clone to results with no gaps.

---

## Overview

This project implements several pipelines for **video activity recognition**, based on Keren (2003) and extensions:

| Pipeline      | Classes | Resolution | Description |
|---------------|---------|------------|-------------|
| **Baseline**  | 2       | 64×64      | Naive Bayes, reproduces Keren 2003 |
| **Improved**  | 2       | 128×128    | Same as Baseline, higher resolution |
| **Multiclass**| 3       | 128×128    | Adds HELLO class (hand_wave_hello) |
| **Deep Learning** | 3  | —          | 3D CNN (R2Plus1D-18) |

**Class labels:** `hand_wave_hello`, `hand_wave_side`, `walking` (Baseline/Improved use only the last two).

---

## Prerequisites

- **Python 3.7+** (3.8+ recommended)
- **Git** (to clone the repo)
- **~2GB free space** for environment and dataset (videos not in repo)

---

## 1. Clone and enter the project

```bash
git clone <REPO_URL>
cd image-processing-project
```

Replace `<REPO_URL>` with your actual repository URL.

---

## 2. Virtual environment and dependencies

```bash
python -m venv venv
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset (required for training and evaluation)

Videos are **not** stored in the repo (large files). To get them:

1. **Download** the slides ZIP from the link below.  
2. **Extract** the ZIP into the **`data/videos`** folder of this project.

**Dataset download (ZIP):**

```text
https://www.dropbox.com/scl/fi/2qds6o9r239y8f61jyv13/Dataset.zip?rlkey=zfaj39gprxweyn7fghd8iw46z&st=uy1vlpwm&dl=0
```

Use “Download” or “Direct download” on the Dropbox page, then unzip into `data/videos`.

See **[data/videos/README - DOWNLOAD DATASET.md](data/videos/README - DOWNLOAD DATASET.md)** for more documentation and the same link.

### After placing videos

- The first time you run a pipeline, the code will split the dataset into **train / validation / test** (e.g. 70% / 15% / 15%) and write CSVs under `data/metadata/` (`train_labels.csv`, `val_labels.csv`, `test_labels.csv`).
- No need to run a separate script; the pipeline scripts use `data/prepare_dataset.py` when needed.

See **[data/README.md](data/README.md)** for scripts (`split_dataset.py`, `prepare_videos.py`) and optional manual splitting.

---

## 4. Run the project (main interface)

From the **project root** (where `run_pipeline.py` is):

```bash
python run_pipeline.py
```

You will get:

1. **Pipeline selection:** Baseline, Improved, Multiclass, Deep Learning.
2. **Mode selection:**
   - **Full pipeline:** train → evaluate → visualize (use this for first run).
   - **Evaluation + visualization:** run on existing model (e.g. on `data/unseen_videos/`).

Follow the prompts. Training and evaluation results are written to the pipeline’s results folder (see below).

### Run a single pipeline by script

```bash
# Baseline
python src/orchestration/run_pipeline_baseline.py

# Improved
python src/orchestration/run_pipeline_improved.py

# Multiclass
python src/orchestration/run_pipeline_multiclass.py

# Deep Learning
python src/orchestration/run_pipeline_deep_learning.py
```

### Run all pipelines and comparison

```bash
python src/orchestration/run_all_pipelines.py
```

Then generate the final comparison report:

```bash
python src/orchestration/generate_final_comparison.py
```

Outputs go to `results/comparison/` (e.g. `comparison_metrics.csv`, `all_pipelines_comparison.txt`).

---

## 5. Where results are saved

| Pipeline       | Results directory (project root)   |
|----------------|-------------------------------------|
| Baseline       | `results_baseline/` (or `results/baseline/` if that exists) |
| Improved       | `results_improved/`                 |
| Multiclass     | `results_multiclass/`               |
| Deep Learning  | `results_deep_learning/`            |
| Comparison     | `results/comparison/`               |

Each pipeline directory typically contains:

- Trained model: `classifier.pkl` or `model.pth`, `training_config.pkl`, `label_mapping.pkl`
- Evaluation: `plots/`, `per_video_breakdown.csv`, `per_class_metrics.csv`, `confusion_matrix_detailed.csv`, `metrics_summary.txt`
- Visualizations: `visualizations/*.mp4` (color-coded block overlay videos)

Heavy outputs (e.g. `.mp4`, `.pkl`, `.pth`) are listed in [.gitignore](.gitignore) so they are not committed.

---

## 6. Presentation / slides

Project slides are **not** stored in the repo (large files). To get them:

1. **Download** the slides ZIP from the link below.  
2. **Extract** the ZIP into the **`docs`** folder of this project (so that slide files sit under `docs/`).

**Slides download (ZIP):**

```text
https://www.dropbox.com/scl/fi/mursqwk8i4pnff4e3ia8z/Presentataion.zip?rlkey=b78y75vpf0qygoyl2p3iwk5je&st=huhuq9yg&dl=0
```

Use “Download” or “Direct download” on the Dropbox page, then unzip into `docs/`.

See **[docs/README - SLIDES.md](docs/README - SLIDES.md)** for more documentation and the same link.

---

## 7. Testing on new (unseen) videos

1. Put videos in `data/unseen_videos/` by class:
   - `data/unseen_videos/hand_wave_hello/`
   - `data/unseen_videos/hand_wave_side/`
   - `data/unseen_videos/walking/`
2. Run:
   ```bash
   python run_pipeline.py
   ```
3. Choose the pipeline and then **“Evaluation + Visualization”** so it uses the existing model and the unseen videos.

---

## 8. Project structure (summary)

```
image-processing-project/
├── README.md                 ← This file
├── run_pipeline.py           ← Main entry (interactive menu)
├── requirements.txt
├── .gitignore                ← Excludes videos, models, heavy PDFs/ZIPs
├── code/                     ← Core modules (feature extraction, classifier, etc.)
├── data/
│   ├── README.md             ← Dataset layout, scripts, download notes
│   ├── metadata/             ← train/val/test CSV labels (generated)
│   ├── videos/               ← Your videos (not in git): hand_wave_hello/, hand_wave_side/, walking/
│   ├── unseen_videos/        ← Optional: extra test videos by class
│   ├── prepare_dataset.py    ← Used by pipelines for split
│   ├── split_dataset.py      ← Standalone split script
│   └── prepare_videos.py     ← Resize, extract frames, etc.
├── src/
│   ├── README.md             ← Pipeline layout and commands
│   ├── baseline/             ← Baseline pipeline scripts
│   ├── improved/
│   ├── multiclass/
│   ├── deep_learning/
│   ├── orchestration/        ← run_pipeline_*.py, run_all_pipelines, generate_final_comparison
│   └── utils/
├── results/                  ← comparison/ and optional baseline subdir
├── results_baseline/         ← Created when running Baseline
├── results_improved/
├── results_multiclass/
├── results_deep_learning/
└── docs/
    ├── README.md             ← Documentation index + slides link
    ├── README - SLIDES.md    ← Slides download only
    ├── ai_assistance_log.md  ← Log of AI-assisted work (course requirement)
    ├── BASELINE_METHODOLOGY.md
    ├── BASELINE_PAPER_COMPARISON.md
    └── ... (other docs)
```

---

## 9. Documentation index

| Document | Description |
|----------|-------------|
| [data/videos/README - DOWNLOAD DATASET.md](data/README%20-%20DOWNLOAD DATASET.md) | Dataset layout, download, scripts, CSV format |
| [docs/README.md](docs/README.md) | Docs index, slides link, baseline methodology |
| [docs/README - SLIDES.md](docs/README%20-%20SLIDES.md) | Slides download link only |
| [src/README.md](src/README.md) | Pipelines and orchestration commands |
| [results/README.md](results/README.md) | What is stored in results directories |
| [docs/BASELINE_METHODOLOGY.md](docs/BASELINE_METHODOLOGY.md) | Baseline implementation details |
| [docs/BASELINE_PAPER_COMPARISON.md](docs/BASELINE_PAPER_COMPARISON.md) | Comparison with Keren 2003 |

---

## 10. Requirements (summary)

- Python 3.7+
- OpenCV, NumPy, SciPy, pandas, scikit-learn, matplotlib, seaborn
- PyTorch and torchvision for the Deep Learning pipeline

See [requirements.txt](requirements.txt) for pinned versions.

---

## 11. References

- Keren, D. (2003). Recognizing image "style" and activities in video using local features and naive Bayes. *Pattern Recognition Letters*, 24(16), 2913–2922.

---

## 12. License

This project is for academic use only, in line with the course guidelines.
