# Project Documentation

This directory contains documentation for the image processing project.

---

## Documents

| Document | Description |
|----------|-------------|
| **BASELINE_METHODOLOGY.md** | Baseline implementation (Keren 2003): training, evaluation, alignment with the paper. |
| **BASELINE_PAPER_COMPARISON.md** | Comparison of the implementation with the paper. |
| **BASELINE_VERIFICATION_REPORT.md** | Verification report for the baseline. |
| **PAPER_ALIGNMENT_CHECK.md** | Paper alignment checklist. |
| **IMPLEMENTATION_ANALYSIS.md** | Implementation analysis. |
| **ai_assistance_log.md** | Log of AI-assisted work (as required by the course). |
| **README - SLIDES.md** | Slides download link only. |

---

## Project slides (presentation)

Slides are **not** stored in the repository (large files). To get them:

1. **Download** the slides ZIP from the link below.  
2. **Extract** the ZIP into this folder (`docs/`).

**Slides download (ZIP):**

```text
https://www.dropbox.com/scl/fi/mursqwk8i4pnff4e3ia8z/Presentataion.zip?rlkey=b78y75vpf0qygoyl2p3iwk5je&st=huhuq9yg&dl=0
```

Use “Download” or “Direct download” on the Dropbox page, then unzip into `docs/`.

See also the main [README.md](../README.md) (section “Presentation / slides”).

---

## Project structure (reference)

- **`/code/`** — Core implementation (feature extraction, classifier, video processing).
- **`/data/`** — Dataset layout, metadata, preprocessing scripts (see [data/README.md](../data/README.md)).
- **`/src/`** — Pipelines (baseline, improved, multiclass, deep learning) and orchestration (see [src/README.md](../src/README.md)).
- **`/results/`** — Comparison outputs; pipeline results are under `results_baseline/`, `results_improved/`, etc. (see [results/README.md](../results/README.md)).
- **`/docs/`** — This directory (documentation).

---

## Quick start (from project root)

Run the main interface from the **project root** (where `run_pipeline.py` is):

```bash
python run_pipeline.py
```

Or run a single pipeline:

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

For full setup (clone, venv, dataset, run), see the main [README.md](../README.md).

---

## Methodology details

For baseline methodology and paper comparison, see:

- [BASELINE_METHODOLOGY.md](BASELINE_METHODOLOGY.md)  
- [BASELINE_PAPER_COMPARISON.md](BASELINE_PAPER_COMPARISON.md)
