# Celestial Objects Classification (CatBoost + Optuna)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-orange.svg)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-3.x-9cf.svg)](https://optuna.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A robust pipeline for **multi-class classification of celestial objects**  
(stars / quasars / galaxies / white dwarfs / red giants / exoplanet candidates) with:

- **Rare-class aware K-Fold** stratification
- **Joint hyperparameter & bias tuning** via **Optuna**
- **Temperature scaling** and **segment-aware class biases** (global / extragalactic / stellar)
- **TTA** (photometric jitter) and a **meta-blender** (multinomial Logistic Regression)
- Optional **GPU** acceleration (CatBoost), automatic CPU fallback by flag

> **Note**: This repository **does not include** any dataset or pretrained weights.  
> Bring your own CSVs and train locally. Artifacts (models, logs, etc.) are **not** committed.

---

## What’s inside

- `main.py` – the complete training/inference script (CLI, Optuna, TTA, physics-aware rules).
- `README.md` – this document.

That’s it. Minimal by design.

---

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# dependencies
pip install "catboost>=1.2" "optuna>=3.0" "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" "joblib>=1.3"


Expected data schema (BYOD)

train.csv: object_id, <features...>, type

type is the target label column (if named differently, the script tries to auto-detect).

test.csv: object_id, <features...>

sample_submission.csv: object_id,<class1>,...,<classN>

Features can include photometry (u_mag..z_mag), SNRs (snr_*), spectral lines (h_alpha_strength, oIII_strength, …),
and astrometric signals (parallax, pm_ra, pm_dec). Missing optional columns are handled gracefully.

The script will:

align train/test columns,

compute derived features (colors, log-SNR, reduced proper motion, extinction-corrected magnitudes, etc.),

apply light physics-aware rules when relevant inputs exist.

Tiny example of sample_submission.csv (inline, not a file):

object_id,exoplanet_candidate,galaxy,main_sequence_star,quasar,red_giant,white_dwarf
OBJ0001,0,1,0,0,0,0
OBJ0002,0,0,1,0,0,0
OBJ0003,0,0,0,1,0,0

Usage

Replace paths with yours. Artifacts are written locally (e.g., models/) and are not part of the repo.

Train from scratch (produce local weights + predictions)
python main.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --sample path/to/sample_submission.csv \
  --out predictions.csv \
  --save_models \
  --device gpu   # or: --device cpu

Inference using your previously saved local weights
python main.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --sample path/to/sample_submission.csv \
  --out predictions.csv \
  --load_models \
  --device gpu   # or: --device cpu

Notable CLI parameters

--save_models — save final full-data CatBoost model and meta-layer locally.
--load_models — load previously saved models/model_full.cbm and models/meta_lr.joblib.
--blend_final — blend weight between the K×seeds ensemble and the final full-data model (0..1, default 0.5).
--device — gpu (default) or cpu (use cpu if no CUDA is available).

How it works (high-level)

Feature engineering – colors, log-SNR, reduced proper motion H_g, extinction-corrected magnitudes, etc.
Rare-class aware K-Fold – stable validation when minority classes exist.
Optuna search – tunes CatBoost HPs + temperature + three bias maps (global/extragal/stellar).
Seed×Fold ensemble – out-of-fold probabilities are used to train a meta-blender (LogReg).
Test-time augmentation – photometric jitter averaging (optional, on by default).
Post-processing – temperature scaling, segment biases, physics-aware nudges, tie-breakers.
Submission – one-hot prediction table matching sample_submission.csv schema.
Primary metric: Macro-F1 (also per-class F1 and confusion insights when you add logging).

Reproducibility tips

Pin versions (pip freeze > requirements.lock) and keep a fixed random_seed.
Save Optuna best trial parameters and thresholds if you need to re-use them across runs.
If your labels set changes, align sample_submission.csv columns accordingly.

Troubleshooting

No GPU? Use --device cpu. The pipeline remains the same (just slower).
Different target name? The script attempts to auto-detect; otherwise rename to type.
Missing features (e.g., parallax, pm_*)? The pipeline still runs; physics-aware rules degrade gracefully.
