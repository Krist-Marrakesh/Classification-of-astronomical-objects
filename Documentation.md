## AstroCatBoostNet: A Technical Guide to the Implementation

### 1. Overview

This document provides a detailed technical breakdown of the `AstroCatBoostNet` implementation, a sophisticated machine learning pipeline designed for astronomical object classification. The script leverages a hybrid approach, combining the predictive power of gradient boosting with domain-specific astrophysical knowledge to achieve high accuracy and physically plausible results.

The core of the system is a **seed- and fold-ensembled CatBoost model**. Its hyperparameters, along with a complex post-processing calibration layer, are jointly optimized using the **Optuna** framework. The final prediction is the result of a multi-stage pipeline involving **Test-Time Augmentation (TTA)**, **temperature scaling**, **segment-specific biases**, a **meta-learning layer**, and **explicit physical rules**.

### 2. The Core Pipeline: From Data to Submission

The script executes a sequential pipeline, which can be broken down into two main phases: **Training & Optimization** and **Inference**.

#### 2.1. Feature Engineering (`add_features`)

The first step is to enrich the raw data with features that have strong predictive power in astrophysics. This function is crucial for exposing the underlying patterns to the model.

- **Color Indices**: Calculates standard astronomical colors (e.g., `u-g`, `g-r`), which are essential for distinguishing different types of stars and galaxies. It automatically generates these features if they are not already present (e.g., `u_g_auto`).
- **Log-Transformed Signal-to-Noise Ratio (SNR)**: Applies a `log1p` transformation to all SNR columns to handle their typically skewed distribution and stabilize variance.
- **Normalized Spectral Strengths**: Normalizes key spectral line strengths (`H-alpha`, `[O III]`, `Na D`) by the `background_noise`. This helps to isolate the signal's intrinsic strength from measurement noise.
- **Proper Motion**: Calculates the total proper motion (`pm_total`) from its right ascension (`pm_ra`) and declination (`pm_dec`) components. This is a key indicator of whether an object is nearby (stellar) or distant (extragalactic).
- **Astrophysical Magnitudes**:
    - **Absolute Magnitude (`absMag_g`)**: Estimates the intrinsic brightness of an object by correcting its apparent magnitude (`g_mag`) for distance (derived from `parallax`) and interstellar extinction.
    - **Reduced Proper Motion (`H_g`)**: A proxy for absolute magnitude that combines apparent magnitude and proper motion, useful for separating stellar populations like dwarfs and giants.
- **Nonlinear Interactions**: Creates interaction terms between adjacent color indices (e.g., `u_g_auto * g_r_auto`) to allow the model to capture more complex relationships in color space.

#### 2.2. Training and Optimization Phase

This phase is dedicated to finding the optimal model configuration and training the final ensemble.

**Step 1: Data Preparation & Stratification**
- The script automatically identifies the object ID and target columns.
- To handle class imbalance and ensure that rare classes are present in every validation set, it uses a custom `make_rare_stratified_folds` function. This is critical for robustly evaluating the model's performance on classes like exoplanet candidates.

**Step 2: Joint Optimization with Optuna (`objective`)**
This is the most innovative part of the script. Instead of tuning the model and calibration steps separately, it performs a joint optimization to find a globally optimal configuration. The `objective` function for Optuna trains a full K-fold CatBoost model for each trial and evaluates its macro F1-score after applying a calibration layer.

The parameters being tuned simultaneously are:
- **CatBoost Hyperparameters**: `depth`, `learning_rate`, `l2_leaf_reg`, `border_count`, `random_strength`, and bootstrap settings (`subsample` or `bagging_temperature`).
- **Temperature Scaling (`temp`)**: A single float value (`T`) used to sharpen or soften the model's probability distributions (`p^T`), making it more or less confident.
- **Segment-Specific Biases (`bias_*`)**: Three independent sets of per-class biases are optimized:
    1.  `bias_global_*`: For objects with ambiguous physical properties.
    2.  `bias_extrag_*`: For objects identified as extragalactic (low proper motion, low parallax).
    3.  `bias_stellar_*`: For objects identified as stellar (high proper motion or high parallax).

**Step 3: Final Ensemble Training**
- After Optuna finds the best set of parameters, the script trains the final model.
- It employs a **double-ensemble** strategy:
    1.  **Seed Ensemble**: The entire training process is repeated with multiple random seeds (`seed_list`). This ensures the final model is robust and not sensitive to a specific random initialization.
    2.  **K-Fold Ensemble**: Within each seed run, a K-fold model is trained.
- This results in a total of `len(seed_list) * folds` individual CatBoost models.

**Step 4: Meta-Model Training**
- During the training of the *first seed*, the Out-of-Fold (OOF) predictions are saved. These are predictions made on the validation data for each fold.
- These OOF predictions, after being calibrated with the best temperature and biases, are used as features to train a final meta-model (`LogisticRegression`).
- The meta-features also include key physical parameters (`pm_total`, `parallax`, `H_g`, `absMag_g`), allowing the meta-model to learn final corrections based on both the base model's output and the physical data.

#### 2.3. Inference Pipeline

The inference phase applies a sequence of transformations to the test data to produce the final classification.

**Step 1: Base Prediction & TTA**
- The test set predictions are averaged across all `S x K` models in the ensemble.
- **Test-Time Augmentation (TTA)** is performed by creating `tta_n` slightly modified versions of the test set. The `jitter_mags` function adds small random noise to the magnitude columns.
- Predictions are made for each jittered dataset and averaged. The final base prediction is a 50/50 blend of the original and TTA-averaged predictions.

**Step 2: Calibration**
- The averaged probabilities are calibrated using the optimal parameters found by Optuna:
    1.  **Temperature Scaling**: The probabilities are raised to the power of `best_temp`.
    2.  **Segment Biases**: The test set is segmented into `stellar`, `extragalactic`, and `global` populations using `make_segment_masks`. The corresponding optimized biases are multiplicatively applied to the probabilities.

**Step 3: Meta-Model Blending**
- The calibrated scores are fed into the trained `LogisticRegression` meta-model to get a second set of predictions.
- The final scores are a weighted average (`alpha=0.7`) of the meta-model's predictions and the calibrated base model's scores. This blending combines the fine-grained power of the meta-model with the robust base ensemble.

**Step 4: Physical Regularization & Tie-Breaking**
- **Soft Physics (`apply_physical_rules`)**: A set of multiplicative rules are applied. For example, objects with extragalactic properties have their probabilities for stellar classes slightly reduced and galactic classes slightly increased.
- **Hard Physics**: A hard rule significantly reduces the probability of `galaxy` or `quasar` for objects with extremely high proper motion or parallax, which are almost certainly stars.
- **Galaxy vs. Quasar Tie-Breaker**: If the model's top two predictions are `galaxy` and `quasar` and the margin between them is very small (`< 0.02`), the script checks for strong emission lines (`H-alpha` or `[O III]`). If present, the `quasar` probability is boosted, otherwise the `galaxy` probability is favored. This embeds a common astrophysical heuristic directly into the pipeline.

**Step 5: Final Submission**
- The class with the highest final score is chosen as the prediction.
- The script generates a one-hot encoded submission file in the required format.

### 3. How to Use the Script

1.  **Dependencies**: Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy scikit-learn catboost optuna
    ```
2.  **Configuration**: Set the global variables at the top of the script:
    - `train_path`, `test_path`, `sample_path`: Paths to your data files.
    - `out_path`: Path for the output submission file.
    - `folds`, `iters`, `seed`, etc.: Adjust these to control the training process. Increasing `tune_trials` or `iters` may yield better results at the cost of longer runtime.
3.  **Execution**: Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```
    The script will print the best hyperparameters found by Optuna and the location of the saved submission file.

### 4. Conclusion

The AstroCatBoostNet script is a powerful and well-structured example of a modern machine learning solution. Its strength lies in its **hybrid design**, which does not rely solely on a black-box model but instead integrates deep domain knowledge at multiple stages: feature engineering, model calibration, and post-processing. The use of **joint optimization** with Optuna ensures that all components of the model work in harmony, while the extensive **ensembling and augmentation** techniques provide robustness and state-of-the-art accuracy.
