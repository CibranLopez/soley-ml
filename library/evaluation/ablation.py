"""
library.evaluation.ablation
==============================

Feature ablation: measure the value of each feature group.

run_feature_ablation(df, output_dir, cfg)
    Trains four RF detectors on different feature subsets:
      1. SCADA only
      2. SCADA + stress (deployment-ready)
      3. SCADA + device physics
      4. Full (SCADA + physics + stress)

    Logs AUC deltas, saves the deployment model, and saves a bar chart.
"""

import logging
import time
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix

log = logging.getLogger("library")


def run_feature_ablation(
    df,
    output_dir: Path,
    cfg,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict]:
    """Feature ablation study for fault detection.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (daytime, features already engineered).
    output_dir : Path
    cfg : BatchConfig
    (RF hyperparameters)

    Returns
    -------
    list[dict]
        One entry per feature set:
        ``{feature_set, n_features, auc}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    log.info("")
    log.info("=" * 65)
    log.info("  FEATURE ABLATION")
    log.info("=" * 65)

    y        = df["fault_active"].values.astype(int)
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=test_size,
        random_state=random_state, stratify=y,
    )

    eng = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
           "performance_ratio", "pr_deviation", "dc_ac_power_ratio",
           "power_step"]
    scada_eng   = [c for c in list(cfg.scada_features) + eng
                   if c in df.columns]
    scada_str   = [c for c in scada_eng + list(cfg.stress_features)
                   if c in df.columns]
    scada_dev   = [c for c in scada_eng + list(cfg.device_features)
                   if c in df.columns]
    full        = [c for c in
                   scada_eng + list(cfg.device_features) + list(cfg.stress_features)
                   if c in df.columns]

    feature_sets = {
        "SCADA only":                   scada_eng,
        "SCADA + stress (deployment)":  scada_str,
        "SCADA + device physics":       scada_dev,
        "Full (SCADA + physics + stress)": full,
    }

    ablation_results = []
    deployment_model = None
    deployment_feats = None

    for set_name, feats in feature_sets.items():
        X, used = build_feature_matrix(df, feats)
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        log.info("  %-45s  %2d feat  AUC=%.4f  (%.1fs)",
                 set_name, len(used), auc, elapsed)

        ablation_results.append({
            "feature_set": set_name,
            "n_features":  len(used),
            "auc":         auc,
        })

        if "deployment" in set_name:
            deployment_model = model
            deployment_feats = used

    # Log delta
    delta = ablation_results[-1]["auc"] - ablation_results[0]["auc"]
    log.info("  Delta (Full − SCADA only): %+.4f AUC", delta)

    # Save deployment model
    if deployment_model is not None:
        import joblib
        dep_path = output_dir / "rf_deployment_detection.pkl"
        joblib.dump({
            "model":         deployment_model,
            "feature_names": deployment_feats,
            "class_names":   ["Healthy", "Faulted"],
            "note":          "Deployment model: SCADA + stress features only. "
                             "Works on real SCADA data (no device physics).",
        }, dep_path)
        log.info("  Saved deployment model: %s  (%d features)",
                 dep_path.name, len(deployment_feats))

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = [r["feature_set"] for r in ablation_results]
    aucs   = [r["auc"]         for r in ablation_results]
    colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]

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
    out = output_dir / "feature_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved feature_ablation.png")

    return ablation_results
