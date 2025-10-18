import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


train_path  = -
test_path   = -
sample_path = -
out_path    = -


folds        = 5
iters        = 1200        
tune_iters   = 400         
tune_trials  = 120         # number of Optuna trays
seed         = 42
seed_list    = [seed, seed+7, seed+13]  # seed ensemble (can be expanded)
exoplanet_boost = 1.6
bias_low     = 0.6
bias_high    = 1.6
tta_n        = 3           # amount of TTA jitters


# Feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # auto-color indexes if missing
    def add_delta(a, b, name):
        if a in df.columns and b in df.columns and name not in df.columns:
            df[name] = df[a] - df[b]
    add_delta("u_mag","g_mag","u_g_auto")
    add_delta("g_mag","r_mag","g_r_auto")
    add_delta("r_mag","i_mag","r_i_auto")
    add_delta("i_mag","z_mag","i_z_auto")
    add_delta("u_mag","r_mag","u_r_auto")
    add_delta("g_mag","i_mag","g_i_auto")

    # log(SNR)
    for col in list(df.columns):
        if str(col).startswith("snr_") and f"log_{col}" not in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            df[f"log_{col}"] = np.log1p(np.clip(x, 0, None))

    # spectral normalization on background_noise
    if "background_noise" in df.columns:
        noise = 1.0 + df["background_noise"].abs()
        for sp in ["h_alpha_strength","oIII_strength","na_d_strength"]:
            if sp in df.columns and f"{sp}_norm" not in df.columns:
                df[f"{sp}_norm"] = df[sp] / noise

    # proper motion module
    if "pm_ra" in df.columns and "pm_dec" in df.columns and "pm_total" not in df.columns:
        df["pm_total"] = np.sqrt(df["pm_ra"].fillna(0)**2 + df["pm_dec"].fillna(0)**2)

    # extinction-corrected g
    if "g_mag" in df.columns and "extinction" in df.columns and "g_mag_corr" not in df.columns:
        df["g_mag_corr"] = df["g_mag"] - 0.8 * df["extinction"].fillna(0)

    # Absolute magnitude in g (with parallax)
    if "g_mag" in df.columns and "parallax" in df.columns:
        plx = pd.to_numeric(df["parallax"], errors="coerce")
        with np.errstate(divide='ignore', invalid='ignore'):
            d_pc = 1000.0 / plx
            M_g = df["g_mag"] - 5*np.log10(np.clip(d_pc, 1e-6, None)) + 5 - df.get("extinction", 0)*0.8
        df["absMag_g"] = pd.Series(M_g).replace([np.inf, -np.inf], np.nan)

    # Reduced proper motion
    if "g_mag" in df.columns and "pm_total" in df.columns:
        pm = pd.to_numeric(df["pm_total"], errors="coerce").clip(lower=1e-6)
        df["H_g"] = df["g_mag"] + 5*np.log10(pm) + 5

    # Nonlinear color interactions
    for a,b in [("u_g_auto","g_r_auto"),("g_r_auto","r_i_auto"),("r_i_auto","i_z_auto")]:
        if a in df.columns and b in df.columns and f"{a}__{b}" not in df.columns:
            df[f"{a}__{b}"] = df[a] * df[b]

    return df


# Utilities
def find_object_and_target(train: pd.DataFrame):
    obj_candidates = [c for c in train.columns if "object" in c.lower() and "id" in c.lower()]
    object_id = obj_candidates[0] if obj_candidates else train.columns[0]
    target_col = None
    for nm in ["target","class","label","y","category","type"]:
        if nm in train.columns:
            target_col = nm; break
    if target_col is None:
        for c in train.columns:
            if c == object_id: continue
            if train[c].dtype == "object" or str(train[c].dtype).startswith("category"):
                if 2 <= train[c].nunique(dropna=True) <= 20:
                    target_col = c; break
    return object_id, target_col

def cat_feature_indices(X: pd.DataFrame):
    cats = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    return [X.columns.get_loc(c) for c in cats], cats

def make_rare_stratified_folds(y: pd.Series, n_splits: int, seed_: int):
    rng = np.random.RandomState(seed_)
    labels = y.astype(str).values
    classes, counts = np.unique(labels, return_counts=True)
    if counts.min() < n_splits:
        n_splits = max(2, int(counts.min()))
    per_class_chunks = {}
    for cls in classes:
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        per_class_chunks[cls] = np.array_split(idx, n_splits)
    folds_idx = []
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        val_idx = np.concatenate([per_class_chunks[c][k] for c in classes]) if len(classes) else np.array([], int)
        val_idx = np.array(sorted(val_idx))
        train_mask = np.ones(len(y), dtype=bool); train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        folds_idx.append((train_idx, val_idx))
    return folds_idx, n_splits

def compute_class_weights(y: pd.Series, exoplanet_boost_: float):
    vc = y.value_counts()
    base = {cls: len(y)/(len(vc)*cnt) for cls, cnt in vc.items()}
    if "exoplanet_candidate" in base:
        base["exoplanet_candidate"] *= exoplanet_boost_
    classes = sorted(vc.index.tolist())
    weights = [base[c] for c in classes]
    return classes, weights

def make_segment_masks(df: pd.DataFrame):
    pm  = pd.to_numeric(df.get("pm_total", 0), errors="coerce").fillna(0)
    plx = pd.to_numeric(df.get("parallax", 0), errors="coerce").fillna(0)
    is_extragal = (pm < 0.3) & (plx.abs() < 0.1)
    is_stellar  = (pm > 5.0) | (plx.abs() > 1.0)
    return is_extragal.values, is_stellar.values

def apply_physical_rules(scores_df: pd.DataFrame, meta_df: pd.DataFrame, class_cols: list):
    out = scores_df.copy()
    has_pm   = "pm_total" in meta_df.columns
    has_plx  = "parallax" in meta_df.columns
    has_bn   = "background_noise" in meta_df.columns
    if has_pm:  pm = meta_df["pm_total"].fillna(0)
    if has_plx: plx = meta_df["parallax"].fillna(0)
    if has_bn:  bn  = meta_df["background_noise"].fillna(meta_df["background_noise"].median())

    # extragalactic-like
    if has_pm and has_plx:
        mask_extragal = (pm < 0.3) & (plx.abs() < 0.1)
        star_cols = [c for c in class_cols if c in ["main_sequence_star","red_giant","white_dwarf"]]
        gal_cols  = [c for c in class_cols if c in ["galaxy","quasar"]]
        if star_cols and gal_cols and mask_extragal.any():
            out.loc[mask_extragal, star_cols] *= 0.8
            out.loc[mask_extragal, gal_cols]  *= 1.1

    # stellar-like
    if has_pm and has_plx:
        mask_stellar = (pm > 5.0) | (plx.abs() > 1.0)
        gal_cols = [c for c in class_cols if c in ["galaxy","quasar"]]
        if gal_cols and mask_stellar.any():
            out.loc[mask_stellar, gal_cols] *= 0.75

    # with a high background - in favor of Galaxy
    if has_bn and "galaxy" in class_cols:
        out.loc[bn > bn.median(), "galaxy"] *= 1.05

    return out

def jitter_mags(df, sigma=0.02, cols=("u_mag","g_mag","r_mag","i_mag","z_mag")):
    dfj = df.copy()
    for c in cols:
        if c in dfj.columns:
            noise = np.random.normal(0, sigma, size=len(dfj))
            dfj[c] = pd.to_numeric(dfj[c], errors="coerce") + noise
    return add_features(dfj)


# MAIN
def main():
    from catboost import CatBoostClassifier, Pool as CBPool
    import optuna

    # download
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    object_id, target_col = find_object_and_target(train)
    if "type" in train.columns:
        target_col = "type"  # explicit commit if the label is called 'type'

    # FE
    train_fe = add_features(train)
    test_fe  = add_features(test)

 # Aligning columns and shaping X/y
    features = [c for c in train_fe.columns if c not in [target_col, object_id]]
    missing_in_test = [c for c in features if c not in test_fe.columns]
    for c in missing_in_test:
        test_fe[c] = np.nan
    features = [c for c in features if c in test_fe.columns]
    X = train_fe[features].copy()
    X_test = test_fe[features].copy()
    y = train_fe[target_col].astype(str)

  # Categorical indices
    cat_idx, _ = cat_feature_indices(X)
    classes, class_weights = compute_class_weights(y, exoplanet_boost)

    # Segments (on X and X_test)
    seg_extrag_X, seg_stellar_X = make_segment_masks(X)
    seg_extrag_T, seg_stellar_T = make_segment_masks(X_test)

   # Folds (rare stratification)
    folds_idx, k = make_rare_stratified_folds(y, folds, seed)

  # Basic CatBoost settings (GPU, early stopping)
    DEFAULT = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=seed,
        early_stopping_rounds=30,
        verbose=False,
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.9,
        allow_writing_files=False,
        grow_policy="SymmetricTree",
    )

    # Joint-Optuna: CatBoost HP + temperature + three sets of biases
    def trial_to_params_and_biases(trial):
        params = dict(
            **DEFAULT,
            iterations=tune_iters,
            depth=trial.suggest_int("depth", 5, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
            border_count=trial.suggest_int("border_count", 128, 255),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        )
        # guarantee correct bootstrap (GPU)
        if trial.suggest_categorical("use_subsample", [True, False]):
            params.update(dict(
                bootstrap_type="Bernoulli",
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
            ))
        else:
            params.update(dict(
                bootstrap_type="Bayesian",
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            ))

        temp = trial.suggest_float("temp", 0.8, 1.4)
        biases_global  = {c: trial.suggest_float(f"bias_{c}",        bias_low, bias_high) for c in classes}
        biases_extrag  = {c: trial.suggest_float(f"bias_extrag_{c}", 0.6, 1.8)            for c in classes}
        biases_stellar = {c: trial.suggest_float(f"bias_stellar_{c}",0.6, 1.8)            for c in classes}
        return params, temp, biases_global, biases_extrag, biases_stellar

    def objective(trial):
        params, temp, b_global, b_extrag, b_stellar = trial_to_params_and_biases(trial)
        oof_proba = np.zeros((len(X), len(classes)), dtype=float)

        for tr_idx, va_idx in folds_idx:
            tr_pool = CBPool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
            va_pool = CBPool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
            model = CatBoostClassifier(**params, class_weights=class_weights)
            model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

            p = model.predict_proba(va_pool)
            mc = list(model.classes_)
            aligned = np.zeros((len(va_idx), len(classes)))
            for j, cls in enumerate(classes):
                aligned[:, j] = p[:, mc.index(cls)]
            oof_proba[va_idx] = aligned

        # temperature scaling
        scaled = oof_proba ** temp
        scaled /= np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)

    # segment masks (on X)
        ex_mask = seg_extrag_X
        st_mask = seg_stellar_X
        df_mask = ~(ex_mask | st_mask)

        for j, cls in enumerate(classes):
            scaled[df_mask, j] *= float(b_global[cls])
            scaled[ex_mask, j] *= float(b_extrag[cls])
            scaled[st_mask, j] *= float(b_stellar[cls])

        preds = np.array(classes)[scaled.argmax(axis=1)]
        return f1_score(y.values, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=tune_trials, show_progress_bar=False)
    best = study.best_params

    best_params = dict(**DEFAULT, iterations=iters)
    bias_map_global, bias_map_extrag, bias_map_stellar = {}, {}, {}
    best_temp = best.get("temp", 1.0)

    for k_, v in best.items():
        if k_.startswith("bias_extrag_"):
            bias_map_extrag[k_.replace("bias_extrag_", "")] = v
        elif k_.startswith("bias_stellar_"):
            bias_map_stellar[k_.replace("bias_stellar_", "")] = v
        elif k_.startswith("bias_"):
            bias_map_global[k_.replace("bias_", "")] = v
        elif k_ in ("temp", "use_subsample"):
            continue
        else:
            best_params[k_] = v

   # guarantee correct bootstrap (GPU)
    bt = best_params.get("bootstrap_type")
    if bt == "Bernoulli":
        best_params.setdefault("subsample", 0.8)
        best_params.pop("bagging_temperature", None)
    elif bt == "Bayesian":
        best_params.setdefault("bagging_temperature", 1.0)
        best_params.pop("subsample", None)
    else:
        best_params["bootstrap_type"] = "Bayesian"
        best_params.setdefault("bagging_temperature", 1.0)
        best_params.pop("subsample", None)

    print("[Best HP]", {
        "depth": best_params.get("depth"),
        "learning_rate": best_params.get("learning_rate"),
        "l2_leaf_reg": best_params.get("l2_leaf_reg"),
        "border_count": best_params.get("border_count"),
        "bootstrap_type": best_params.get("bootstrap_type"),
        "subsample": best_params.get("subsample"),
        "bagging_temperature": best_params.get("bagging_temperature"),
        "random_strength": best_params.get("random_strength"),
    })
    print("[Best temp]", round(best_temp, 3))
    print("[Bias global]",  {k: round(v,3) for k,v in bias_map_global.items()})
    print("[Bias extrag]",  {k: round(v,3) for k,v in bias_map_extrag.items()})
    print("[Bias stellar]", {k: round(v,3) for k,v in bias_map_stellar.items()})

  
  # Final training: seed ensemble × K-fold
# + parallel construction of OOF (from the first seed) for the meta layer
   
    all_models = []
    oof_proba_meta = np.zeros((len(X), len(classes)), dtype=float)
    first_seed_models = None

    for si, s in enumerate(seed_list):
        params_seed = dict(best_params, random_seed=s)
        models = []
        for tr_idx, va_idx in folds_idx:
            tr_pool = CBPool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
            va_pool = CBPool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
            m = CatBoostClassifier(**params_seed, class_weights=class_weights)
            m.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            models.append(m)

            if si == 0:
                p = m.predict_proba(va_pool)
                mc = list(m.classes_)
                aligned = np.zeros((len(va_idx), len(classes)))
                for j, cls in enumerate(classes):
                    aligned[:, j] = p[:, mc.index(cls)]
              # averaging over fold models: save, then divide by K
                oof_proba_meta[va_idx] += aligned

        all_models.append(models)
        if si == 0:
            first_seed_models = models

  # average over K (for the first seed)
    oof_counts = np.zeros(len(X), dtype=int)
    for _, va_idx in folds_idx:
        oof_counts[va_idx] += 1
    oof_proba_meta /= np.clip(oof_counts[:, None], 1, None)

    # temperature + segment-biases on OOF
    oof_scores = oof_proba_meta ** best_temp
    oof_scores /= np.clip(oof_scores.sum(axis=1, keepdims=True), 1e-12, None)
    ex_mask = seg_extrag_X
    st_mask = seg_stellar_X
    df_mask = ~(ex_mask | st_mask)
    for j, cls in enumerate(classes):
        oof_scores[df_mask, j] *= float(bias_map_global.get(cls, 1.0))
        oof_scores[ex_mask, j] *= float(bias_map_extrag.get(cls, 1.0))
        oof_scores[st_mask, j] *= float(bias_map_stellar.get(cls, 1.0))

   # meta-features (OOF-score + physical cues)
    def num_col(dff, name):
        return pd.to_numeric(dff.get(name, 0), errors="coerce").fillna(0).values.reshape(-1,1)
    meta_X = np.hstack([
        oof_scores,
        num_col(train_fe, "pm_total"),
        np.abs(num_col(train_fe, "parallax")),
        num_col(train_fe, "H_g"),
        num_col(train_fe, "absMag_g"),
    ])
    meta_y = y.values

    meta = LogisticRegression(multi_class="multinomial", max_iter=200, C=2.0, n_jobs=-1)
    meta.fit(meta_X, meta_y)

   
  # Inference: average over all models → TTA → temperature → segment-biases
# → physics (soft and hard) → metalayer-blend → tie-break → argmax
    
    test_pool = CBPool(X_test, cat_features=cat_idx)
    proba_test = np.zeros((len(X_test), len(classes)), dtype=float)

    for models in all_models:
        for m in models:
            p = m.predict_proba(test_pool)
            mc = list(m.classes_)
            for j, cls in enumerate(classes):
                proba_test[:, j] += p[:, mc.index(cls)]
    proba_test /= (len(all_models) * len(all_models[0]))

  # TTA: Photometry Jitter and Averaging
    tta_scores = np.zeros_like(proba_test)
    for _ in range(tta_n):
        tta_fe = jitter_mags(test, sigma=0.02)
        # align columns
        for c in features:
            if c not in tta_fe.columns:
                tta_fe[c] = np.nan
        tta_X = tta_fe[features]
        tta_pool = CBPool(tta_X, cat_features=cat_idx)
        tmp = np.zeros_like(proba_test)
        for models in all_models:
            for m in models:
                p = m.predict_proba(tta_pool)
                mc = list(m.classes_)
                for j, cls in enumerate(classes):
                    tmp[:, j] += p[:, mc.index(cls)]
        tmp /= (len(all_models) * len(all_models[0]))
        tta_scores += tmp
    tta_scores /= max(1, tta_n)
    proba_test = 0.5 * proba_test + 0.5 * tta_scores

    # temperature
    scores = proba_test ** best_temp
    scores /= np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)

 # three sets of biases by segments (on X_test)
    ex_mask_T = seg_extrag_T
    st_mask_T = seg_stellar_T
    df_mask_T = ~(ex_mask_T | st_mask_T)
    for j, cls in enumerate(classes):
        scores[df_mask_T, j] *= float(bias_map_global.get(cls, 1.0))
        scores[ex_mask_T, j] *= float(bias_map_extrag.get(cls, 1.0))
        scores[st_mask_T, j] *= float(bias_map_stellar.get(cls, 1.0))

  # meta-features on the test and blend
    meta_T = np.hstack([
        scores,
        num_col(test_fe, "pm_total"),
        np.abs(num_col(test_fe, "parallax")),
        num_col(test_fe, "H_g"),
        num_col(test_fe, "absMag_g"),
    ])
    meta_proba = meta.predict_proba(meta_T)
    alpha = 0.7
    scores_blend = alpha * meta_proba + (1 - alpha) * scores

    # soft physics
    scores_df = pd.DataFrame(scores_blend, columns=classes)
    scores_df = apply_physical_rules(scores_df, test_fe, classes)

 # hard stellar clip for extreme pm/parallax
    pmv = pd.to_numeric(test_fe.get("pm_total", 0), errors="coerce").fillna(0)
    plx = pd.to_numeric(test_fe.get("parallax", 0), errors="coerce").fillna(0)
    hard_stellar = (pmv > 20) | (plx.abs() > 3)
    if "galaxy" in classes and "quasar" in classes:
        g_idx = classes.index("galaxy")
        q_idx = classes.index("quasar")
        S = scores_df.values
        S[hard_stellar.values, g_idx] *= 0.2
        S[hard_stellar.values, q_idx] *= 0.2
        scores_df = pd.DataFrame(S, columns=classes)

    # tie-breaker: galaxy vs quasar
    if "galaxy" in classes and "quasar" in classes:
        g_idx = classes.index("galaxy")
        q_idx = classes.index("quasar")
        S = scores_df.values
        top = S.max(axis=1, keepdims=True)
        second = np.partition(S, -2, axis=1)[:, -2][:, None]
        margin = (top - second).ravel()
        near = margin < 0.02  # draw threshold

        ha = pd.to_numeric(test_fe.get("h_alpha_strength_norm", test_fe.get("h_alpha_strength", 0)), errors="coerce").fillna(0)
        o3 = pd.to_numeric(test_fe.get("oIII_strength_norm",  test_fe.get("oIII_strength",  0)), errors="coerce").fillna(0)
        thr_ha = np.nanquantile(ha, 0.8) if np.isfinite(ha).any() else 0
        thr_o3 = np.nanquantile(o3, 0.8) if np.isfinite(o3).any() else 0
        strong_lines = (ha > thr_ha) | (o3 > thr_o3)

        mask = near & strong_lines.values
        S[mask, q_idx] *= 1.05
        S[mask, g_idx] *= 0.95
        scores_df = pd.DataFrame(S, columns=classes)

    final_labels = np.array(classes)[scores_df.values.argmax(axis=1)]

    
    # One-hot submission
    class_cols = [c for c in sample.columns if c != "object_id"]
    if not class_cols:
        class_cols = classes

    # determine object_id from test
    obj_candidates = [c for c in test.columns if "object" in c.lower() and "id" in c.lower()]
    obj_col = obj_candidates[0] if obj_candidates else test.columns[0]

    sub = pd.DataFrame({"object_id": test[obj_col]})
    onehot = pd.DataFrame(0, index=np.arange(len(test)), columns=class_cols)
    lc_map = {c.lower(): c for c in class_cols}
    for i, lbl in enumerate(final_labels):
        if lbl in onehot.columns:
            onehot.loc[i, lbl] = 1
        elif lbl.lower() in lc_map:
            onehot.loc[i, lc_map[lbl.lower()]] = 1
        else:
            onehot.iloc[i, 0] = 1

    sub = pd.concat([sub, onehot.astype(int)], axis=1)
    sub.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

