"""
library.models.random_forest
==============================

scikit-learn RandomForestClassifier wrappers for:

  train_rf_detection(df, feature_names, cfg, ...)
      Binary fault detection (healthy vs. faulted).
      Returns the trained model, hold-out ROC-AUC, and CV scores.

  train_rf_classification(df, feature_names, cfg, ...)
      Multi-class fault type classification.
      Returns the trained model, hold-out AUC, CV scores, and LabelEncoder.

Both functions optionally save the model to disk with joblib.
"""

import logging
import time
from pathlib import Path

import numpy as np

from library.config import BatchConfig
from library.features import build_feature_matrix

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Detection
# ---------------------------------------------------------------------------

def train_rf_detection(
    df,
    feature_names: list[str],
    cfg: BatchConfig,
    *,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 20,
    test_size: float = 0.2,
    cv: int = 5,
    save_path: Path | None = None,
    random_state: int = 42,
):
    """Train a Random Forest for binary fault detection.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (daytime, features already engineered).
    feature_names : list[str]
    cfg : BatchConfig
    n_estimators, max_depth, min_samples_leaf : RF hyperparameters
    test_size : float
    cv : int
        Number of stratified CV folds.
    save_path : Path or None
        If given, saves ``{model, feature_names, class_names}`` with joblib.
    random_state : int

    Returns
    -------
    model : RandomForestClassifier
    auc : float  — hold-out ROC-AUC
    cv_scores : np.ndarray  — per-fold AUC scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split

    log.info("")
    log.info("=" * 65)
    log.info("  RF FAULT DETECTION (binary)")
    log.info("=" * 65)

    X, used = build_feature_matrix(df, feature_names)
    y = df["fault_active"].values.astype(int)

    n_healthy = (y == 0).sum()
    n_faulted = (y == 1).sum()
    log.info("  Healthy: %s  (%.1f%%)", f"{n_healthy:,}",
             100.0 * n_healthy / len(y))
    log.info("  Faulted: %s  (%.1f%%)", f"{n_faulted:,}",
             100.0 * n_faulted / len(y))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )

    log.info("  Training RF (%d trees) …", n_estimators)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    log.info("")
    log.info(classification_report(y_te, y_pred,
                                   target_names=["Healthy", "Faulted"]))
    auc = roc_auc_score(y_te, y_prob)
    log.info("  ROC AUC (hold-out): %.4f", auc)

    if cv > 1:
        log.info("  Running %d-fold CV …", cv)
        cv_scores = cross_val_score(model, X, y, cv=cv,
                                    scoring="roc_auc", n_jobs=1)
        log.info("  CV ROC AUC: %.4f ± %.4f  (%s)",
                 cv_scores.mean(), cv_scores.std(),
                 ", ".join(f"{s:.4f}" for s in cv_scores))
    else:
        cv_scores = np.array([auc])

    if save_path is not None:
        import joblib
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model,
                     "feature_names": used,
                     "class_names": ["Healthy", "Faulted"]}, save_path)
        log.info("  Saved: %s", save_path)

    return model, auc, cv_scores


# ---------------------------------------------------------------------------
#  Classification
# ---------------------------------------------------------------------------

def train_rf_classification(
    df,
    feature_names: list[str],
    cfg: BatchConfig,
    *,
    n_estimators: int = 200,
    max_depth: int = 20,
    min_samples_leaf: int = 10,
    test_size: float = 0.2,
    cv: int = 5,
    save_path: Path | None = None,
    save_deployment_path: Path | None = None,
    random_state: int = 42,
):
    """Train a Random Forest for multi-class fault type classification.

    Only faulted rows are used (``fault_active == True``).

    Parameters
    ----------
    df : pd.DataFrame
    feature_names : list[str]
    cfg : BatchConfig
    save_path : Path or None
        Save full model (all features).
    save_deployment_path : Path or None
        Save SCADA + stress deployment model (works on real SCADA data).
    (other params as in train_rf_detection)

    Returns
    -------
    model : RandomForestClassifier
    auc : float or None — weighted OvR ROC-AUC
    cv_scores : np.ndarray
    le : LabelEncoder
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  RF FAULT CLASSIFICATION (multi-class)")
    log.info("=" * 65)

    faulted = df[df["fault_active"]].copy()
    n_classes = faulted["fault_type"].nunique()
    log.info("  %s faulted rows, %d fault types",
             f"{len(faulted):,}", n_classes)

    if n_classes < 2:
        log.info("  Not enough fault types — skipping.")
        from sklearn.preprocessing import LabelEncoder
        return None, None, None, LabelEncoder()

    vc = faulted["fault_type"].value_counts()
    for ft, cnt in vc.items():
        log.info("  %-30s %10s", ft, f"{cnt:,}")

    X, used = build_feature_matrix(faulted, feature_names)
    le = LabelEncoder()
    y_enc = le.fit_transform(faulted["fault_type"].values)
    class_names = list(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state,
        stratify=y_enc)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced",
        random_state=random_state, n_jobs=-1,
    )

    log.info("  Training RF (%d trees, %d classes) …", n_estimators, n_classes)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred    = le.inverse_transform(model.predict(X_te))
    y_te_lab  = le.inverse_transform(y_te)

    log.info("")
    log.info(classification_report(y_te_lab, y_pred))

    try:
        y_prob = model.predict_proba(X_te)
        auc    = roc_auc_score(y_te, y_prob, multi_class="ovr",
                               average="weighted")
        log.info("  Weighted ROC AUC (OvR): %.4f", auc)
    except Exception:
        auc = None

    if cv > 1:
        log.info("  Running %d-fold CV …", cv)
        cv_scores = cross_val_score(model, X, y_enc, cv=cv,
                                    scoring="roc_auc_ovr_weighted", n_jobs=1)
        log.info("  CV ROC AUC: %.4f ± %.4f  (%s)",
                 cv_scores.mean(), cv_scores.std(),
                 ", ".join(f"{s:.4f}" for s in cv_scores))
    else:
        cv_scores = np.array([auc]) if auc else np.array([0.0])

    if save_path is not None:
        import joblib
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "label_encoder": le,
                     "feature_names": used, "class_names": class_names},
                    save_path)
        log.info("  Saved: %s", save_path)

    # --- Deployment model (SCADA + stress only) ---
    if save_deployment_path is not None:
        eng = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
               "performance_ratio", "pr_deviation",
               "dc_ac_power_ratio", "power_step"]
        deploy_feats = (list(cfg.scada_features) + eng
                        + list(cfg.stress_features))
        deploy_feats = [c for c in deploy_feats if c in faulted.columns]

        X_dep, dep_used = build_feature_matrix(faulted, deploy_feats)
        X_dtr, X_dte, y_dtr, y_dte = train_test_split(
            X_dep, y_enc, test_size=test_size,
            random_state=random_state, stratify=y_enc)

        m_dep = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        log.info("  Training DEPLOYMENT classification model …")
        m_dep.fit(X_dtr, y_dtr)

        import joblib
        save_deployment_path = Path(save_deployment_path)
        save_deployment_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": m_dep, "label_encoder": le,
            "feature_names": dep_used, "class_names": class_names,
            "note": "Deployment model: SCADA + stress features only.",
        }, save_deployment_path)
        log.info("  Saved deployment model: %s", save_deployment_path)

    return model, auc, cv_scores, le
