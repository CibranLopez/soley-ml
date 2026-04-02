"""
SOLEY ML Data Factory -- Fault Detection & Classification Demo
===============================================================

Demonstrates that SOLEY's first-principles synthetic PV data produces
ML-trainable fault signatures at 5-minute resolution.

Five demonstrations:
  1. Fault Detection      -- Is the system healthy or faulted? (random split)
  2. Fault Classification  -- Which of the 11 fault types is active?
  3. Temporal Robustness   -- Train on years 1-2, test on year 3
  4. Generalization Test   -- Train on 3 sites, test on the 4th
  5. Feature Ablation      -- SCADA-only vs Full: value of SOLEY physics

Auto-configures from batch_config.json and parquet schema.
Works with ANY SOLEY batch output -- no hardcoded values.

Usage:
    python ml_fault_demo.py                          # full demo
    python ml_fault_demo.py --data-dir /path/to/data # custom path
    python ml_fault_demo.py --quick                  # 1 location only
    python ml_fault_demo.py --scada-only             # SCADA features only
    python ml_fault_demo.py --no-subsample           # use all rows

Requirements:
    pip install scikit-learn
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_utils import (
    BatchConfig, add_features, load_data_rf,
    plot_confusion, plot_importance,
)

log = logging.getLogger("soley_ml")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ===================================================================
#  MODEL TRAINING & EVALUATION
# ===================================================================

def build_feature_matrix(df, feature_names):
    """Extract feature matrix X from DataFrame."""
    cols = [c for c in feature_names if c in df.columns]
    missing = set(feature_names) - set(cols)
    if missing:
        log.info("  (missing columns filled with 0: %s)", ", ".join(missing))

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
    return X, cols


def run_detection(df, feature_names, output_dir, cfg):
    """TASK 1: Binary fault detection."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split

    log.info("")
    log.info("=" * 65)
    log.info("  TASK 1: BINARY FAULT DETECTION (random split)")
    log.info("=" * 65)

    X, used_features = build_feature_matrix(df, feature_names)
    y = df["fault_active"].values.astype(int)

    n_healthy = (y == 0).sum()
    n_faulted = (y == 1).sum()
    log.info("  Healthy:  %10s  (%5.1f%%)", f"{n_healthy:,}",
             100.0 * n_healthy / len(y))
    log.info("  Faulted:  %10s  (%5.1f%%)", f"{n_faulted:,}",
             100.0 * n_faulted / len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    log.info("  Training Random Forest (200 trees) ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred, target_names=["Healthy", "Faulted"],
    )
    auc = roc_auc_score(y_test, y_prob)

    log.info("")
    log.info(report)
    log.info("  ROC AUC (holdout): %.4f", auc)

    log.info("  Running 5-fold stratified cross-validation ...")
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring="roc_auc", n_jobs=1,
    )
    log.info("  CV ROC AUC: %.4f +/- %.4f  (folds: %s)",
             cv_scores.mean(), cv_scores.std(),
             ", ".join(f"{s:.4f}" for s in cv_scores))

    plot_confusion(y_test, y_pred, ["Healthy", "Faulted"],
                   "Fault Detection", output_dir / "confusion_detection.png")
    plot_importance(model, used_features, "Fault Detection",
                    output_dir / "importance_detection.png",
                    cfg.scada_features, cfg.device_features)

    import joblib
    joblib.dump({
        "model": model,
        "feature_names": used_features,
        "class_names": ["Healthy", "Faulted"],
    }, output_dir / "rf_model_detection.pkl")
    log.info("  Saved trained model: rf_model_detection.pkl")

    return model, auc, cv_scores


def run_classification(df, feature_names, output_dir, cfg):
    """TASK 2: Multi-class fault classification."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  TASK 2: FAULT TYPE CLASSIFICATION")
    log.info("=" * 65)

    faulted = df[df["fault_active"]].copy()
    n_classes = faulted["fault_type"].nunique()
    log.info("  %s faulted rows, %d fault types", f"{len(faulted):,}",
             n_classes)

    if n_classes < 2:
        log.info("  Not enough fault types -- skipping.")
        return None, None, None

    log.info("")
    vc = faulted["fault_type"].value_counts()
    for ft, cnt in vc.items():
        log.info("  %-30s %10s", ft, f"{cnt:,}")

    X, used_features = build_feature_matrix(faulted, feature_names)
    y = faulted["fault_type"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    log.info("  Training Random Forest (200 trees, %d classes) ...",
             n_classes)
    t0 = time.time()
    model.fit(X_train, y_train)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    report = classification_report(y_test_labels, y_pred_labels)
    log.info("")
    log.info(report)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr",
                            average="weighted")
        log.info("  Weighted ROC AUC (OvR): %.4f", auc)
    except Exception:
        auc = None

    log.info("  Running 5-fold stratified cross-validation ...")
    cv_scores = cross_val_score(
        model, X, y_enc, cv=5, scoring="roc_auc_ovr_weighted", n_jobs=1,
    )
    log.info("  CV ROC AUC: %.4f +/- %.4f  (folds: %s)",
             cv_scores.mean(), cv_scores.std(),
             ", ".join(f"{s:.4f}" for s in cv_scores))

    plot_confusion(y_test_labels, y_pred_labels, class_names,
                   "Fault Classification",
                   output_dir / "confusion_classification.png")
    plot_importance(model, used_features, "Fault Classification",
                    output_dir / "importance_classification.png",
                    cfg.scada_features, cfg.device_features)

    import joblib
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "feature_names": used_features,
        "class_names": class_names,
    }, output_dir / "rf_model_classification.pkl")
    log.info("  Saved trained model: rf_model_classification.pkl")

    # Train DEPLOYMENT classification model (SCADA + stress only)
    # This is the model that works on real SCADA data (no Voc/Jsc/FF)
    engineered = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                  "performance_ratio", "pr_deviation",
                  "dc_ac_power_ratio", "power_step"]
    deploy_features = (list(cfg.scada_features) + engineered
                       + list(cfg.stress_features))
    deploy_features = [c for c in deploy_features if c in faulted.columns]

    X_dep, dep_used = build_feature_matrix(faulted, deploy_features)
    X_dep_train, X_dep_test, y_dep_train, y_dep_test = train_test_split(
        X_dep, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )

    model_dep = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    log.info("")
    log.info("  Training DEPLOYMENT classification model "
             "(SCADA + stress, %d features) ...", len(dep_used))
    t0 = time.time()
    model_dep.fit(X_dep_train, y_dep_train)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_dep_pred = model_dep.predict(X_dep_test)
    dep_report = classification_report(
        le.inverse_transform(y_dep_test),
        le.inverse_transform(y_dep_pred),
    )
    log.info("")
    log.info(dep_report)

    try:
        y_dep_prob = model_dep.predict_proba(X_dep_test)
        dep_auc = roc_auc_score(y_dep_test, y_dep_prob, multi_class="ovr",
                                average="weighted")
        log.info("  DEPLOYMENT Classification AUC: %.4f", dep_auc)
    except Exception:
        dep_auc = None

    plot_confusion(
        le.inverse_transform(y_dep_test),
        le.inverse_transform(y_dep_pred),
        class_names,
        "Fault Classification (DEPLOYMENT: SCADA+stress)",
        output_dir / "confusion_classification_deployment.png",
    )
    plot_importance(model_dep, dep_used,
                    "Fault Classification (DEPLOYMENT)",
                    output_dir / "importance_classification_deployment.png",
                    cfg.scada_features, cfg.device_features)

    joblib.dump({
        "model": model_dep,
        "label_encoder": le,
        "feature_names": dep_used,
        "class_names": class_names,
        "note": "DEPLOYMENT model: uses only SCADA + stress features. "
                "Stress labels computed from weather via condition_scanner.py.",
    }, output_dir / "rf_deployment_classification.pkl")
    log.info("  Saved DEPLOYMENT model: rf_deployment_classification.pkl")

    return model, auc, cv_scores


def run_temporal(df, feature_names, output_dir, cfg):
    """TASK 3: Temporal split -- train on years 1-2, test on year 3."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  TASK 3: TEMPORAL SPLIT (train years 1-2, test year 3)")
    log.info("=" * 65)

    if "sim_year" not in df.columns:
        log.info("  sim_year column missing -- skipping.")
        return {}

    years = sorted(df["sim_year"].unique())
    if len(years) < 2:
        log.info("  Only %d year(s) -- skipping temporal test.", len(years))
        return {}

    # Train on all but last year, test on last year
    train_years = years[:-1]
    test_year = years[-1]

    train_df = df[df["sim_year"].isin(train_years)]
    test_df = df[df["sim_year"] == test_year]
    log.info("  Train: %s rows (years %s)", f"{len(train_df):,}",
             ", ".join(str(y) for y in train_years))
    log.info("  Test:  %s rows (year %d)", f"{len(test_df):,}", test_year)

    results = {}

    # Detection
    log.info("")
    log.info("  --- Detection (binary) ---")
    X_train, _ = build_feature_matrix(train_df, feature_names)
    X_test, _ = build_feature_matrix(test_df, feature_names)
    y_train = train_df["fault_active"].values.astype(int)
    y_test = test_df["fault_active"].values.astype(int)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc_det = roc_auc_score(y_test, y_prob)

    report = classification_report(
        y_test, y_pred, target_names=["Healthy", "Faulted"],
    )
    log.info("")
    log.info(report)
    log.info("  Temporal Detection AUC: %.4f", auc_det)
    results["detection_auc"] = auc_det

    plot_confusion(y_test, y_pred, ["Healthy", "Faulted"],
                   f"Temporal Detection (Year {test_year})",
                   output_dir / "confusion_temporal_detection.png")

    # Classification
    log.info("")
    log.info("  --- Classification (multi-class) ---")
    train_faulted = train_df[train_df["fault_active"]]
    test_faulted = test_df[test_df["fault_active"]]

    if test_faulted["fault_type"].nunique() < 2:
        log.info("  Not enough fault types in test year -- skipping.")
        return results

    le = LabelEncoder()
    le.fit(df[df["fault_active"]]["fault_type"].values)

    X_train_f, _ = build_feature_matrix(train_faulted, feature_names)
    X_test_f, _ = build_feature_matrix(test_faulted, feature_names)
    y_train_f = le.transform(train_faulted["fault_type"].values)
    y_test_f = le.transform(test_faulted["fault_type"].values)

    model_cls = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model_cls.fit(X_train_f, y_train_f)

    y_pred_f = model_cls.predict(X_test_f)
    y_pred_labels = le.inverse_transform(y_pred_f)
    y_test_labels = le.inverse_transform(y_test_f)

    report_cls = classification_report(y_test_labels, y_pred_labels)
    log.info("")
    log.info(report_cls)

    try:
        y_prob_f = model_cls.predict_proba(X_test_f)
        auc_cls = roc_auc_score(y_test_f, y_prob_f, multi_class="ovr",
                                average="weighted")
        log.info("  Temporal Classification AUC: %.4f", auc_cls)
        results["classification_auc"] = auc_cls
    except Exception:
        pass

    plot_confusion(y_test_labels, y_pred_labels, list(le.classes_),
                   f"Temporal Classification (Year {test_year})",
                   output_dir / "confusion_temporal_classification.png")

    return results


def run_generalization(df, feature_names, output_dir, cfg):
    """TASK 4: Cross-location generalization (leave-one-site-out)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  TASK 4: CROSS-LOCATION GENERALIZATION")
    log.info("=" * 65)

    df = df.copy()
    df["loc_id"] = df.apply(
        lambda r: cfg.get_loc_id(r["latitude"], r["longitude"]), axis=1,
    )
    locations = sorted(df["loc_id"].unique())

    if len(locations) < 2:
        log.info("  Need >= 2 locations. Skipping.")
        return []

    # Fit label encoder on all faulted data
    all_faulted = df[df["fault_active"]]
    le = LabelEncoder()
    if len(all_faulted) > 0 and all_faulted["fault_type"].nunique() >= 2:
        le.fit(all_faulted["fault_type"].values)
        do_classification = True
    else:
        do_classification = False

    results = []
    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        train_df = df[df["loc_id"].isin(train_locs)]
        test_df = df[df["loc_id"] == test_loc]

        # Get human-readable names
        test_lat, test_lon = [float(x) for x in test_loc.split("_")]
        name = cfg.get_location_name(test_lat, test_lon)
        train_names = ", ".join(
            cfg.get_location_name(
                *[float(x) for x in loc.split("_")]
            ) for loc in train_locs
        )

        # Detection
        X_train, _ = build_feature_matrix(train_df, feature_names)
        X_test, _ = build_feature_matrix(test_df, feature_names)
        y_train = train_df["fault_active"].values.astype(int)
        y_test = test_df["fault_active"].values.astype(int)

        model_det = RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model_det.fit(X_train, y_train)

        y_prob = model_det.predict_proba(X_test)[:, 1]
        try:
            auc_det = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc_det = float("nan")

        # Classification
        auc_cls = float("nan")
        if do_classification:
            train_faulted = train_df[train_df["fault_active"]]
            test_faulted = test_df[test_df["fault_active"]]

            if test_faulted["fault_type"].nunique() >= 2:
                X_train_f, _ = build_feature_matrix(train_faulted, feature_names)
                X_test_f, _ = build_feature_matrix(test_faulted, feature_names)
                y_train_f = le.transform(train_faulted["fault_type"].values)
                y_test_f = le.transform(test_faulted["fault_type"].values)

                model_cls = RandomForestClassifier(
                    n_estimators=100, max_depth=15,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )
                model_cls.fit(X_train_f, y_train_f)

                try:
                    y_prob_f = model_cls.predict_proba(X_test_f)
                    auc_cls = roc_auc_score(y_test_f, y_prob_f,
                                            multi_class="ovr", average="weighted")
                except Exception:
                    pass

        report_dict = classification_report(y_test,
                                            model_det.predict(X_test),
                                            output_dict=True)

        log.info("  Train: %-40s  Test: %-20s  Det AUC=%.4f  Cls AUC=%.4f  Acc=%.4f",
                 train_names, name, auc_det, auc_cls, report_dict["accuracy"])

        results.append({
            "test_location": name,
            "test_loc_id": test_loc,
            "detection_auc": auc_det,
            "classification_auc": auc_cls,
            "accuracy": report_dict["accuracy"],
        })

    # Plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names = [r["test_location"] for r in results]
        x = range(len(results))

        ax = axes[0]
        det_aucs = [r["detection_auc"] for r in results]
        bars = ax.bar(x, det_aucs, color="#2563eb", width=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("ROC AUC", fontsize=12)
        ax.set_title("Cross-Location Detection\n(trained on OTHER sites)", fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random")
        ax.legend(fontsize=10)
        for bar, v in zip(bars, det_aucs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

        ax = axes[1]
        cls_aucs = [r["classification_auc"] for r in results]
        bars = ax.bar(x, cls_aucs, color="#10b981", width=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("ROC AUC (OvR weighted)", fontsize=12)
        ax.set_title("Cross-Location Classification\n(trained on OTHER sites)", fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random")
        ax.legend(fontsize=10)
        for bar, v in zip(bars, cls_aucs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

        plt.tight_layout()
        fig.savefig(output_dir / "cross_location.png", dpi=150)
        plt.close(fig)
        log.info("  Saved cross_location.png")

    return results


def run_ablation(df, output_dir, cfg):
    """TASK 5: Feature ablation -- SCADA vs Full."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    log.info("")
    log.info("=" * 65)
    log.info("  TASK 5: FEATURE ABLATION")
    log.info("=" * 65)

    y = df["fault_active"].values.astype(int)
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y,
    )

    # Build feature sets from auto-detected columns + engineered features
    scada_with_eng = list(cfg.scada_features) + [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "performance_ratio",
        "pr_deviation", "dc_ac_power_ratio", "power_step",
    ]
    # Filter to only columns that exist in the DataFrame
    scada_with_eng = [c for c in scada_with_eng if c in df.columns]

    # SCADA + stress = the DEPLOYMENT feature set (works on real SCADA data)
    scada_stress = scada_with_eng + list(cfg.stress_features)
    scada_stress = [c for c in scada_stress if c in df.columns]

    feature_sets = {
        "SCADA only": scada_with_eng,
        "SCADA + stress (deployment)": scada_stress,
        "SCADA + device physics": scada_with_eng + list(cfg.device_features),
        "Full (SCADA + physics + stress)": (
            scada_with_eng + list(cfg.device_features) + list(cfg.stress_features)
        ),
    }

    ablation_results = []
    deployment_model = None
    deployment_features = None

    for set_name, features in feature_sets.items():
        X, used = build_feature_matrix(df, features)
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        log.info("  %-40s  %2d features  AUC=%.4f  (%.1fs)",
                 set_name, len(used), auc, elapsed)

        ablation_results.append({
            "feature_set": set_name,
            "n_features": len(used),
            "auc": auc,
        })

        # Save the deployment model (SCADA + stress)
        if "deployment" in set_name:
            deployment_model = model
            deployment_features = used

    delta = ablation_results[-1]["auc"] - ablation_results[0]["auc"]
    log.info("")
    log.info("  Delta (Full - SCADA only): %+.4f AUC", delta)
    if delta > 0:
        log.info("  --> SOLEY physics features improve fault detection by "
                 "%.2f percentage points.", delta * 100)

    # Save deployment model (the one that works on real SCADA data)
    if deployment_model is not None:
        import joblib
        joblib.dump({
            "model": deployment_model,
            "feature_names": deployment_features,
            "class_names": ["Healthy", "Faulted"],
            "note": "DEPLOYMENT model: uses only SCADA + stress features "
                    "(no device physics). Stress labels can be computed from "
                    "weather data via condition_scanner.py.",
        }, output_dir / "rf_deployment_detection.pkl")
        log.info("  Saved DEPLOYMENT model: rf_deployment_detection.pkl "
                 "(%d features)", len(deployment_features))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [r["feature_set"] for r in ablation_results]
    aucs = [r["auc"] for r in ablation_results]
    colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"][:len(ablation_results)]

    bars = ax.bar(range(len(ablation_results)), aucs, color=colors, width=0.6)
    ax.set_xticks(range(len(ablation_results)))
    ax.set_xticklabels(names, fontsize=10, rotation=5, ha="center")
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_title("Feature Ablation: Value of SOLEY Physics Features\n"
                 "(same data, same split, same model)", fontsize=13)
    min_auc = min(aucs)
    ax.set_ylim(max(0, min_auc - 0.05), 1.02)
    for bar, v, n in zip(bars, aucs, [r["n_features"] for r in ablation_results]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"AUC={v:.4f}\n({n} feat.)",
                ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "feature_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved feature_ablation.png")

    return ablation_results


# ===================================================================
#  REPORT
# ===================================================================

def save_report(data, all_results, cross_results, temporal_results,
                ablation_results, output_dir, args, cfg):
    """Write a human-readable summary."""
    lines = [
        "SOLEY ML Data Factory -- Demonstration Report",
        "=" * 55, "",
        "DATASET",
        "-" * 40,
        f"  Data directory:     {args.data_dir}",
        f"  Total rows:         {len(data):>12,}",
        f"  Locations:          {data['latitude'].nunique():>12}",
        f"  DC/AC ratios:       {data['dc_ac_ratio'].nunique():>12}",
        f"  Fault types:        {data['fault_type'].nunique():>12}",
        f"  Array capacity:     {cfg.array_kwp:>12.2f} kWp",
        f"  Features:           {len(cfg.all_feature_cols):>12} auto-detected",
        "",
    ]

    for r in all_results:
        lines.append(f"{r['task']}:")
        if r.get("auc") is not None:
            lines.append(f"  ROC AUC (holdout):  {r['auc']:.4f}")
        if r.get("cv_mean") is not None:
            lines.append(f"  ROC AUC (5-fold CV): {r['cv_mean']:.4f} "
                         f"+/- {r['cv_std']:.4f}")
        lines.append("")

    if temporal_results:
        lines.append("Temporal Split:")
        for k, v in temporal_results.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    if cross_results:
        lines.append("Cross-Location Generalization:")
        for cr in cross_results:
            lines.append(
                f"  Test: {cr['test_location']:<20s}  "
                f"Det AUC={cr['detection_auc']:.4f}  "
                f"Cls AUC={cr['classification_auc']:.4f}"
            )
        avg_det = np.mean([cr["detection_auc"] for cr in cross_results])
        avg_cls = np.nanmean([cr["classification_auc"] for cr in cross_results])
        lines.append(f"  Average Detection AUC:       {avg_det:.4f}")
        lines.append(f"  Average Classification AUC:  {avg_cls:.4f}")
        lines.append("")

    if ablation_results:
        lines.append("Feature Ablation:")
        for ar in ablation_results:
            lines.append(f"  {ar['feature_set']:<40s}  "
                         f"{ar['n_features']:2d} feat  AUC={ar['auc']:.4f}")
        delta = ablation_results[-1]["auc"] - ablation_results[0]["auc"]
        lines.append(f"  Delta (Full - SCADA only): {delta:+.4f}")
        lines.append("")

    lines += [
        "", "KEY TAKEAWAY", "-" * 40,
        "SOLEY generates first-principles PV data at 5-minute resolution",
        "with per-timestep fault labels. ML models trained on this data",
        "learn physically meaningful signatures that generalize across",
        "locations and time.",
        "",
        "Plots saved to: " + str(output_dir.resolve()),
    ]

    text = "\n".join(lines)
    (output_dir / "demo_report.txt").write_text(text, encoding="utf-8")

    log.info("")
    log.info(text)
    log.info("")
    log.info("All outputs saved to %s", output_dir.resolve())


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SOLEY ML Data Factory -- Fault Detection Demo",
    )
    parser.add_argument("--data-dir", default="batch_output")
    parser.add_argument("--output-dir", default="ml_demo_results")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 1 location only")
    parser.add_argument("--scada-only", action="store_true",
                        help="SCADA features only (no device physics/stress)")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--no-subsample", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Auto-detect configuration from batch output
    log.info("Auto-detecting batch configuration ...")
    cfg = BatchConfig(args.data_dir)

    # Build feature list from auto-detected columns + engineered features
    engineered = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                  "performance_ratio", "pr_deviation",
                  "dc_ac_power_ratio", "power_step"]
    feature_list = list(cfg.scada_features) + engineered
    if not args.scada_only:
        feature_list += cfg.device_features + cfg.stress_features
    log.info("Feature mode: %s (%d features)",
             "SCADA only" if args.scada_only else "Full",
             len(feature_list))

    # Load data
    loc_filter = None
    if args.quick and cfg.locations:
        # Use first location
        first_loc = list(cfg.locations.keys())[0]
        loc_filter = f"{first_loc[0]}_{first_loc[1]}"

    row_limit = None if args.no_subsample else args.max_rows
    data = load_data_rf(args.data_dir, cfg, max_rows=row_limit,
                        location_filter=loc_filter)
    data = add_features(data, cfg.array_kwp)

    all_results = []

    # Task 1
    model_det, auc_det, cv_det = run_detection(
        data, feature_list, output_dir, cfg)
    all_results.append({
        "task": "Binary Fault Detection",
        "auc": auc_det,
        "cv_mean": cv_det.mean(),
        "cv_std": cv_det.std(),
    })

    # Task 2
    model_cls, auc_cls, cv_cls = run_classification(
        data, feature_list, output_dir, cfg)
    if auc_cls is not None:
        all_results.append({
            "task": "Fault Classification",
            "auc": auc_cls,
            "cv_mean": cv_cls.mean() if cv_cls is not None else None,
            "cv_std": cv_cls.std() if cv_cls is not None else None,
        })

    # Task 3
    temporal_results = run_temporal(data, feature_list, output_dir, cfg)

    # Task 4
    if not args.quick:
        cross_results = run_generalization(data, feature_list, output_dir, cfg)
    else:
        log.info("\n  Skipping cross-location (--quick mode)")
        cross_results = []

    # Task 5
    if not args.scada_only:
        ablation_results = run_ablation(data, output_dir, cfg)
    else:
        log.info("\n  Skipping ablation (--scada-only)")
        ablation_results = []

    # Report
    save_report(data, all_results, cross_results, temporal_results,
                ablation_results, output_dir, args, cfg)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        log.error("Install with:  pip install scikit-learn")
        sys.exit(1)
