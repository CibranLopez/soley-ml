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


def _classification_report_dict(y_true, y_pred, class_names=None):
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


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


# ---------------------------------------------------------------------------
#  Registry-based training  (unified data pipeline with PyTorch models)
# ---------------------------------------------------------------------------

def train_rf_from_registry(
    task_name: str,
    target_col: str,
    registry: list[dict],
    feature_cols: list[str],
    cfg: BatchConfig,
    *,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 20,
    max_train_rows: int | None = 1_000_000,
    save_path: Path | None = None,
    save_deployment_path: Path | None = None,
    random_state: int = 42,
    output_dir: Path | None = None,
    artifact_prefix: str | None = None,
    lookback_window: int = 12,
    return_details: bool = False,
) -> tuple | dict:
    """Train a Random Forest using the same file-level registry splits as
    the PyTorch pipeline, enabling apples-to-apples model comparison.

    Unlike :func:`train_rf_detection` / :func:`train_rf_classification`, which
    do a random row-level ``train_test_split`` on the full DataFrame (causing
    temporal leakage across simulation runs), this function:

    * uses the pre-assigned ``"split"`` key in ``registry`` so that all rows
      from a given parquet file are exclusively in train **or** test — never
      both;
    * is called with the identical registry used by :func:`run_pytorch_task`,
      so every model family sees the same train / val / test files.

    Parameters
    ----------
    task_name : str
        Human-readable name, e.g. ``"Fault Detection"``.
    target_col : str
        ``"fault_active"`` (detection) or ``"fault_type"`` (classification).
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` + ``assign_splits``.
    feature_cols : list[str]
    cfg : BatchConfig
    n_estimators, max_depth, min_samples_leaf : RF hyperparameters
    max_train_rows : int or None
        Subsample training rows to manage memory.  ``None`` = no limit.
    save_path : Path or None
        Save ``{model, feature_names, class_names, label_encoder}`` with
        joblib.
    save_deployment_path : Path or None
        If given, also save a deployment model trained on SCADA + stress
        features only (no device physics).
    random_state : int

    Returns
    -------
    model : RandomForestClassifier
    auc : float or None
    le : LabelEncoder or None
    class_names : list[str]
    """
    import time

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    from library.data.loader import load_split  # single canonical loader

    log.info("")
    log.info("=" * 65)
    log.info("  RF %s  (registry-based splits)", task_name.upper())
    log.info("=" * 65)

    is_cls = (target_col == "fault_type")

    train_e = [e for e in registry if e.get("split") == "train"]
    test_e  = [e for e in registry if e.get("split") == "test"]
    if is_cls:
        train_e = [e for e in train_e if e["fault_type"] != "none"]
        test_e  = [e for e in test_e  if e["fault_type"] != "none"]

    log.info("  Files: train=%d  test=%d", len(train_e), len(test_e))

    # load_split calls load_registry_entry for every file so sim_year_filter
    # and lookback_window are applied identically for RF and PyTorch models.
    log.info("  Loading train files …")
    train_df = load_split(train_e, cfg, lookback_window=lookback_window)
    log.info("  Loading test  files …")
    test_df  = load_split(test_e,  cfg, lookback_window=lookback_window)

    # For classification keep only fault-active rows (a file spans healthy
    # baseline + faulted periods; we only classify the faulted ones).
    if is_cls:
        if "fault_active" in train_df.columns:
            train_df = train_df[train_df["fault_active"]].copy()
        if "fault_active" in test_df.columns:
            test_df  = test_df[test_df["fault_active"]].copy()

    if train_df.empty or test_df.empty:
        log.info("  Not enough data — skipping.")
        # Return consistently so run_rf_task never receives an unexpected
        # 4-tuple and falls into a stale dead-code branch.
        return {} if return_details else (None, None, None, [])

    if max_train_rows and len(train_df) > max_train_rows:
        if is_cls and "fault_type" in train_df.columns:
            from sklearn.model_selection import train_test_split as _tts
            try:
                idx, _ = _tts(
                    np.arange(len(train_df)),
                    train_size=max_train_rows,
                    stratify=train_df["fault_type"].values,
                    random_state=random_state,
                )
                train_df = train_df.iloc[idx].reset_index(drop=True)
                log.info("  Stratified subsample → %s rows", f"{len(train_df):,}")
            except ValueError:
                rng = np.random.RandomState(random_state)
                train_df = train_df.sample(n=max_train_rows, random_state=rng)
                log.info("  Random subsample → %s rows (stratify fallback)",
                         f"{len(train_df):,}")
        else:
            rng = np.random.RandomState(random_state)
            train_df = train_df.sample(n=max_train_rows, random_state=rng)
            log.info("  Subsampled train → %s rows", f"{len(train_df):,}")

    X_tr, used = build_feature_matrix(train_df, feature_cols)
    X_te, _    = build_feature_matrix(test_df,  used)

    le = None
    if is_cls:
        le = LabelEncoder()
        y_tr       = le.fit_transform(train_df["fault_type"].values)
        known_test = test_df["fault_type"].isin(le.classes_)
        dropped = int((~known_test).sum())
        if dropped:
            log.info("  Skipping %d test rows with unseen fault types", dropped)
        test_df = test_df.loc[known_test].copy()
        if test_df.empty:
            log.info("  No known fault types left in test split — skipping.")
            return {} if return_details else (None, None, None, [])
        X_te, _    = build_feature_matrix(test_df,  used)
        y_te       = le.transform(test_df["fault_type"].values)
        class_names = list(le.classes_)
        log.info("  Classes (%d): %s", len(class_names), class_names)
    else:
        y_tr        = train_df["fault_active"].values.astype(int)
        y_te        = test_df["fault_active"].values.astype(int)
        class_names = ["Healthy", "Faulted"]
        n_h = int((y_tr == 0).sum())
        n_f = int((y_tr == 1).sum())
        log.info("  Train: %s rows (healthy=%s, faulted=%s)  Test: %s rows",
                 f"{len(X_tr):,}", f"{n_h:,}", f"{n_f:,}", f"{len(X_te):,}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    log.info("  Training RF (%d trees) …", n_estimators)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    if is_cls:
        y_true_out = le.inverse_transform(y_te)
        y_pred_out = le.inverse_transform(y_pred)
        report = _classification_report_dict(y_true_out, y_pred_out)
        log.info("")
        log.info(classification_report(
            y_true_out,
            y_pred_out,
        ))
        try:
            auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = None
    else:
        y_true_out = y_te
        y_pred_out = y_pred
        report = _classification_report_dict(y_true_out, y_pred_out, class_names)
        log.info("")
        log.info(classification_report(y_te, y_pred, target_names=class_names))
        auc = roc_auc_score(y_te, y_prob[:, 1])

    if auc is not None:
        log.info("  ROC AUC (registry test split): %.4f", auc)

    if output_dir is not None:
        from library.visualization import plot_confusion, plot_importance

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = artifact_prefix or task_name.lower().replace(" ", "_")
        plot_confusion(
            y_true_out,
            y_pred_out,
            class_names,
            f"{task_name} — RANDOM_FOREST",
            output_dir / f"confusion_{tag}.png",
        )
        plot_importance(
            model,
            used,
            f"{task_name} — RANDOM_FOREST",
            output_dir / f"importance_{tag}.png",
            scada_features=cfg.scada_features,
            device_features=cfg.device_features,
        )

    if save_path is not None:
        import joblib
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         model,
            "feature_names": used,
            "class_names":   class_names,
            "label_encoder": le,
        }, save_path)
        log.info("  Saved: %s", save_path)

    # Deployment model: SCADA + stress features only (no device physics)
    if save_deployment_path is not None:
        eng = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
               "performance_ratio", "pr_deviation",
               "dc_ac_power_ratio", "power_step"]
        deploy_feats = (list(cfg.scada_features) + eng
                        + list(cfg.stress_features))
        deploy_feats = [c for c in deploy_feats if c in train_df.columns]

        X_dep_tr, dep_used = build_feature_matrix(train_df, deploy_feats)
        X_dep_te, _        = build_feature_matrix(test_df,  dep_used)

        m_dep = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        log.info("  Training deployment model (SCADA+stress only) …")
        m_dep.fit(X_dep_tr, y_tr)

        try:
            dep_prob = m_dep.predict_proba(X_dep_te)
            dep_auc  = (roc_auc_score(y_te, dep_prob[:, 1])
                        if not is_cls else
                        roc_auc_score(y_te, dep_prob,
                                      multi_class="ovr", average="weighted"))
            log.info("  Deployment AUC: %.4f", dep_auc)
        except Exception:
            pass

        import joblib
        save_deployment_path = Path(save_deployment_path)
        save_deployment_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         m_dep,
            "feature_names": dep_used,
            "class_names":   class_names,
            "label_encoder": le,
            "note":          "Deployment model: SCADA + stress features only.",
        }, save_deployment_path)
        log.info("  Saved deployment model: %s", save_deployment_path)

    if return_details:
        return {
            "model": model,
            "report": report,
            "auc": auc,
            "label_encoder": le,
            "class_names": class_names,
            "feature_names": used,
            "y_true": y_true_out,
            "y_pred": y_pred_out,
            "model_path": str(save_path) if save_path is not None else None,
        }

    return model, auc, le, class_names


def run_rf_task(
    task_name: str,
    target_col: str,
    registry: list[dict],
    feature_cols: list[str],
    cfg: BatchConfig,
    *,
    output_dir: Path,
    train_entries: list[dict] | None = None,
    val_entries: list[dict] | None = None,
    test_entries: list[dict] | None = None,
    artifact_prefix: str | None = None,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 20,
    max_train_rows: int | None = 1_000_000,
    random_state: int = 42,
    lookback_window: int = 12,
    return_details: bool = False,
) -> dict:
    """Unified Random Forest runner matching the notebook-facing API.

    Thin wrapper around :func:`train_rf_from_registry` that resolves the
    ``train_entries`` / ``val_entries`` / ``test_entries`` coming from
    :func:`~library.models.pipeline.run_model_task` into a synthetic
    registry and forwards ``lookback_window`` so feature engineering is
    identical to the PyTorch path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_entries is None:
        train_entries = [e for e in registry if e.get("split") == "train"]
    if val_entries is None:
        val_entries   = [e for e in registry if e.get("split") == "val"]
    if test_entries is None:
        test_entries  = [e for e in registry if e.get("split") == "test"]

    synthetic_registry = [dict(e, split="train") for e in train_entries]
    synthetic_registry.extend(dict(e, split="val")  for e in val_entries)
    synthetic_registry.extend(dict(e, split="test") for e in test_entries)

    tag = artifact_prefix or task_name.lower().replace(" ", "_")
    details = train_rf_from_registry(
        task_name,
        target_col,
        synthetic_registry,
        feature_cols,
        cfg,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_train_rows=max_train_rows,
        save_path=output_dir / f"model_{tag}_random_forest.pkl",
        save_deployment_path=output_dir / f"deployment_{tag}_random_forest.pkl",
        random_state=random_state,
        output_dir=output_dir,
        artifact_prefix=f"{tag}_random_forest",
        lookback_window=lookback_window,
        return_details=True,
    )

    # train_rf_from_registry returns {} on empty data (return_details=True
    # is always passed above, so a bare 4-tuple is never returned).
    if not details or not isinstance(details, dict):
        return {}

    return {"random_forest": details if return_details else details["report"]}
