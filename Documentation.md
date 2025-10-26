
### AstroCatBoostNet: A Physically-Regularized Multi-Segment Gradient Ensemble for Astronomical Object Classification

### Abstract

AstroCatBoostNet is a hybrid gradient-boosting architecture for high-precision astrophysical classification. It integrates GPU-accelerated CatBoost ensembles with physically aware calibration, temperature scaling, and segment-specific bias adaptation jointly optimized via Optuna. The model reaches macro-F1 $\approx 0.97$ and provides interpretable, physically constrained decision boundaries distinguishing stellar, galactic, and extragalactic sources.

### 1. Introduction

Modern astronomical surveys (Gaia DR3, SDSS DR16, etc.) produce billions of records, requiring scalable and physically consistent ML systems. Traditional classifiers ignore domain priors such as parallax ($\pi$), proper motion ($\mu$), or interstellar extinction ($A_V$).

AstroCatBoostNet embeds such priors directly into its probabilistic calibration process. The result is a model that is both interpretable and state-of-the-art in accuracy.

### 2. Data Representation

Each object is represented by a feature vector $x_i \in \mathbb{R}^d$ and label $y_i \in \mathcal{Y} = \{1, \dots, K\}$, where $\mathcal{Y} = \{\text{main\_sequence\_star, red\_giant, white\_dwarf, galaxy, quasar, exoplanet\_candidate}\}$.
**Derived features**

$u - g = u_{\text{mag}} - g_{\text{mag}}$

$g - r = g_{\text{mag}} - r_{\text{mag}}$

$r - i = r_{\text{mag}} - i_{\text{mag}}$

$i - z = i_{\text{mag}} - z_{\text{mag}}$

Reduced proper motion:

$$
H_g = g_{\text{mag}} + 5 \cdot \log_{10}(\sqrt{\mu_{\alpha}^2 + \mu_{\delta}^2}) + 5
$$
Absolute magnitude with extinction correction:

$$
M_g = g_{\text{mag}} - 5 \cdot \log_{10}(1000 / \pi) + 5 - 0.8 \cdot A_V
$$

These relations approximate empirical color–motion separations between stellar and extragalactic populations.

### 3. Architecture Overview

AstroCatBoostNet has three key layers:

1.  **Base Gradient-Ensemble** — CatBoost GPU model $f_{\theta}$ minimizing multi-class log-loss.
2.  **Physical Calibration** — applies temperature scaling and segment-wise biases.
3.  **Segment Ensemble + Tie-Breaker** — interprets stellar/extragalactic context and spectral ambiguity.

#### 3.1 Base Gradient-Ensemble

Each CatBoost model outputs logits $z_i = f_{\theta}(x_i)$ and class probabilities:

$$
p_{ik} = \frac{\exp(z_{ik})}{\sum_{j} \exp(z_{ij})}
$$

Ensemble averaging over $S$ seeds and $K$ folds:

$$
\bar{p}_i = \frac{1}{SK} \cdot \sum_{s} \sum_{k} p_{i}^{(s,k)}
$$

Training objective (multi-class cross-entropy):

$$
L(\theta) = -\frac{1}{N} \sum_{i} \sum_{k} y_{ik} \log p_{ik}
$$

#### 3.2 Physical Calibration Layer

Temperature scaling and bias correction:

$$
\tilde{p}_{ik} = \frac{(p_{ik})^T \cdot b_{s(i),k}}{\sum_{j} (p_{ij})^T \cdot b_{s(i),j}}
$$

Where:
- $T$ — temperature controlling probability sharpness
- $b_{s(i),k}$ — bias for segment $s(i)$ and class $k$
- $s(i) \in \{\text{global, stellar, extragalactic}\}$, defined as:
  - $\text{stellar} \quad \text{if} \quad |\pi_i| > 1 \text{ or } \mu_i > 5$
  - $\text{extragalactic} \quad \text{if} \quad |\pi_i| < 0.1 \text{ and } \mu_i < 0.3$
  - $\text{global} \quad \text{otherwise}$


#### 3.3 Segment Tie-Breaker

Margin between top-2 classes:

$$
\Delta_i = \max_k \tilde{p}_{ik} - \text{second\_max}_k \tilde{p}_{ik}
$$

If $\Delta_i < \epsilon$ and emission lines (H$\alpha$ or [O III]) are strong:
→ increase $\tilde{p}(\text{quasar})$ by 5% and decrease $\tilde{p}(\text{galaxy})$ by 5%.


This simulates astrophysical discrimination in spectral surveys.

### 4. Optimization

#### 4.1 Rare-Class Stratification

Each class appears in every fold:

$$
|V_j \cap \{ y_i = c \}| \ge 1 \quad \text{for all } c, j
$$

#### 4.2 Joint Optuna Optimization


Joint search for CatBoost hyperparameters, temperature $T$, and bias $b_{s,k}$:

$$
\text{maximize} \quad F_{1\_macro}(\text{argmax}_k \tilde{p}_{ik})
$$

**Search ranges:**

| Parameter       | Range         |
|-----------------|---------------|
| `depth`         | 5 – 10        |
| `learning_rate` | 0.01 – 0.2    |
| `l2_leaf_reg`   | 0.5 – 50      |
| temperature $T$ | 0.8 – 1.4     |
| bias $b_{s,k}$  | 0.6 – 1.8     |

#### 4.3 Early Stopping

Stop when $\Delta F_1 < 10^{-4}$ over 30 iterations.

### 5. Physical Regularization

Astrophysical priors are applied multiplicatively:

$$
R(x) =
\begin{cases}
0.8  & \text{for stellar classes if } (\pi, \mu) \text{ indicate extragalactic} \\
1.1  & \text{for galaxies/quasars if } (\pi, \mu) \text{ near zero} \\
0.75 & \text{for galaxies/quasars if } |\pi| > 1 \text{ or } \mu > 5 \\
1.05 & \text{for galaxy if } \text{background\_noise} > \text{median}
\end{cases}
$$

Corrected posterior:

$$
\hat{p}_{ik} = R(x_i) \cdot \tilde{p}_{ik}
$$

### 6. Ensemble and Inference

Final prediction distribution per object:

$$
\bar{p}_i = \frac{1}{SK} \sum_{s} \sum_{k} p_{i}^{(s,k)}
$$

Predicted class:

$$
\hat{y}_i = \text{argmax}_k \bar{p}_{ik}
$$

### 7. Results

| Metric | Validation (OOF) | Public Test |
|---|---|---|
| Macro-F₁ | $0.971 \pm 0.004$ | $0.968 – 0.972$ |
| Inference time | $1.1 \text{ ms} / \text{object}$ | RTX 4080 GPU |
| Features used | 48 – 52 | after augmentation|

Residual confusion remains between galaxy and quasar, reflecting intrinsic spectral overlap.

### 9. Conclusion

We presented AstroCatBoostNet, a physically regularized gradient ensemble for astronomical classification. Its unified optimization of CatBoost hyperparameters, temperature scaling, and astrophysical bias vectors provides high accuracy ($F_1 \approx 0.97$) and interpretability.

Formally, inference operates in a constrained posterior space:

$$
\tilde{p}_{ik} = \frac{(p_{ik})^T \cdot b_{s(i),k}}{\sum_{j} (p_{ij})^T \cdot b_{s(i),j}}, \quad \text{with} \quad b_{s(i),k} > 0
$$

This bridges data-driven learning with theory-driven astrophysics, enabling physically consistent predictions.

Future extensions include:

-   differentiable bias learning with uncertainty calibration;
-   spectral autoencoder embeddings;
-   multi-objective optimization balancing physical plausibility and accuracy.

**Acknowledgements**

Implemented using CatBoost 1.2 (GPU) and Optuna 3.0, trained on NVIDIA RTX 4080. Inspired by physical priors from Gaia DR3 and SDSS DR16 datasets.
