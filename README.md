# Celestial Objects Classification (CatBoost + Optuna)

Classification of celestial objects (stars / quasars / galaxies / white dwarfs / red giants / exoplanet candidates) with rare-class stratification, TTA, and meta-blending.

## Features
- CatBoost MultiClass + custom stratification for rare classes.
- Optuna: joint hyperparameter and bias weight tuning (global/extragal/stellar) + temperature.
- Postprocessing with "physical" rules (soft/hard), TTA (photometric jitter).
- Meta-layer (LogisticRegression) based on OOF scores + physical features.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
