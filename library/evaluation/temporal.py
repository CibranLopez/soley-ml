"""
library.evaluation.temporal
=============================

Temporal split evaluation: train on years 1…N-1, test on year N.

run_temporal_split(df, feature_names, output_dir, cfg)
    Runs both detection and classification with a temporal split.
    Saves confusion matrix PNGs and returns a results dict.
"""

import logging
import time
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix

log = logging.getLogger("library")


def run_temporal_split(
    df,
    feature_names: list[str],
    output_dir: Path,
    cfg,
) -> dict:
    """Train on early years, evaluate on the final year.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a ``sim_year`` column.
    feature_names : list[str]
    output_dir : Path
    cfg : BatchConfig

    Returns
    -------
    dict  with keys ``detection_auc``, ``classification_auc`` (if available).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    from library.visualization import plot_confusion

    log.info("")
    log.info("=" * 65)
    log.info("  TEMPORAL SPLIT (train years 1…N-1, test year N)")
    log.info("=" * 65)

    if "sim_year" not in df.columns:
        log.info("  sim_year column missing — skipping.")
        return {}

    years = sorted(df["sim_year"].unique())
    if len(years) < 2:
        log.info("  Only %d year(s) — skipping.", len(years))
        return {}

    train_years = years[:-1]
    test_year   = years[-1]
    train_df    = df[df["sim_year"].isin(train_years)]
    test_df     = df[df["sim_year"] == test_year]

    log.info("  Train: %s rows (years %s)",
             f"{len(train_df):,}", ", ".join(str(y) for y in train_years))
    log.info("  Test:  %s rows (year %d)", f"{len(test_df):,}", test_year)

    results: dict = {}

    # ---- Detection -------------------------------------------------------
    log.info("")
    log.info("  — Detection —")
    X_tr, _ = build_feature_matrix(train_df, feature_names)
    X_te, _ = build_feature_matrix(test_df,  feature_names)
    y_tr    = train_df["fault_active"].values.astype(int)
    y_te    = test_df["fault_active"].values.astype(int)

    det_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    det_model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = det_model.predict(X_te)
    y_prob = det_model.predict_proba(X_te)[:, 1]
    auc_det = roc_auc_score(y_te, y_prob)

    log.info("")
    log.info(classification_report(y_te, y_pred,
                                   target_names=["Healthy", "Faulted"]))
    log.info("  Temporal Detection AUC: %.4f", auc_det)
    results["detection_auc"] = auc_det

    plot_confusion(
        y_te, y_pred, ["Healthy", "Faulted"],
        f"Temporal Detection (Year {test_year})",
        output_dir / "confusion_temporal_detection.png",
    )

    # ---- Classification --------------------------------------------------
    log.info("")
    log.info("  — Classification —")
    train_f = train_df[train_df["fault_active"]]
    test_f  = test_df[test_df["fault_active"]]

    if test_f["fault_type"].nunique() < 2:
        log.info("  Not enough fault types in test year — skipping.")
        return results

    le = LabelEncoder()
    le.fit(df[df["fault_active"]]["fault_type"].values)

    X_tr_f, _ = build_feature_matrix(train_f, feature_names)
    X_te_f, _ = build_feature_matrix(test_f,  feature_names)
    y_tr_f    = le.transform(train_f["fault_type"].values)
    y_te_f    = le.transform(test_f["fault_type"].values)

    cls_model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cls_model.fit(X_tr_f, y_tr_f)

    y_pred_f = cls_model.predict(X_te_f)
    log.info("")
    log.info(classification_report(
        le.inverse_transform(y_te_f),
        le.inverse_transform(y_pred_f),
    ))

    try:
        y_prob_f = cls_model.predict_proba(X_te_f)
        auc_cls  = roc_auc_score(y_te_f, y_prob_f,
                                 multi_class="ovr", average="weighted")
        log.info("  Temporal Classification AUC: %.4f", auc_cls)
        results["classification_auc"] = auc_cls
    except Exception:
        pass

    plot_confusion(
        le.inverse_transform(y_te_f),
        le.inverse_transform(y_pred_f),
        list(le.classes_),
        f"Temporal Classification (Year {test_year})",
        output_dir / "confusion_temporal_classification.png",
    )

    return results
