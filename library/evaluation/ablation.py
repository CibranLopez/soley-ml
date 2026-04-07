"""
library.evaluation.ablation
==============================

Feature ablation: measure the value of each feature group.

run_feature_ablation(registry, output_dir, cfg)
    Trains four RF detectors on different feature subsets using the same
    file-level registry splits as the main training pipeline:
      1. SCADA only
      2. SCADA + stress (deployment-ready)
      3. SCADA + device physics
      4. Full (SCADA + physics + stress)

    Train files → entries with split="train".
    Test  files → entries with split="test".
    This gives exactly the same train/test boundary used by all other models.
"""

import logging
import time
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix, add_features

log = logging.getLogger("library")


def run_feature_ablation(
    registry: list[dict],
    output_dir: Path,
    cfg,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 20,
    max_train_rows: int | None = 600_000,
    random_state: int = 42,
) -> list[dict]:
    """Feature ablation study for fault detection.

    Uses **file-level** train/test splits from ``registry`` so that no
    row from a training run appears in the test set.  This is the same
    split used by :func:`~library.models.random_forest.train_rf_from_registry`
    and :func:`~library.models.trainer.run_pytorch_task`.

    Parameters
    ----------
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` + ``assign_splits``.
        Each entry must have a ``"split"`` key.
    output_dir : Path
    cfg : BatchConfig
    n_estimators, max_depth, min_samples_leaf : RF hyperparameters
    max_train_rows : int or None
        Cap on training rows (random subsample).  ``None`` = no limit.
    random_state : int

    Returns
    -------
    list[dict]
        One entry per feature set: ``{feature_set, n_features, auc}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    log.info("")
    log.info("=" * 65)
    log.info("  FEATURE ABLATION  (registry file-level splits)")
    log.info("=" * 65)

    # ---- Load data from registry splits ----------------------------------
    def _load_entries(entries: list[dict]) -> pd.DataFrame:
        frames = []
        for e in entries:
            df_e = pd.read_parquet(e["path"], columns=cfg.load_cols)
            df_e = add_features(df_e, cfg.array_kwp)
            if len(df_e):
                frames.append(df_e)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    train_entries = [e for e in registry if e.get("split") == "train"]
    test_entries  = [e for e in registry if e.get("split") == "test"]
    log.info("  Loading train (%d files) …", len(train_entries))
    train_df = _load_entries(train_entries)
    log.info("  Loading test  (%d files) …", len(test_entries))
    test_df  = _load_entries(test_entries)

    if train_df.empty or test_df.empty:
        log.info("  Not enough data — skipping.")
        return []

    if max_train_rows and len(train_df) > max_train_rows:
        rng      = np.random.RandomState(random_state)
        y_tmp    = train_df["fault_active"].values.astype(int)
        # Stratified subsample
        from sklearn.model_selection import train_test_split as _tts
        idx, _   = _tts(np.arange(len(train_df)), train_size=max_train_rows,
                        random_state=random_state, stratify=y_tmp)
        train_df = train_df.iloc[idx].reset_index(drop=True)
        log.info("  Subsampled train to %s rows", f"{len(train_df):,}")

    y_tr = train_df["fault_active"].values.astype(int)
    y_te = test_df["fault_active"].values.astype(int)

    log.info("  Train: %s rows  Test: %s rows",
             f"{len(train_df):,}", f"{len(test_df):,}")

    # ---- Feature subsets -------------------------------------------------
    eng = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
           "performance_ratio", "pr_deviation", "dc_ac_power_ratio",
           "power_step"]
    scada_eng  = [c for c in list(cfg.scada_features) + eng
                  if c in train_df.columns]
    scada_str  = [c for c in scada_eng + list(cfg.stress_features)
                  if c in train_df.columns]
    scada_dev  = [c for c in scada_eng + list(cfg.device_features)
                  if c in train_df.columns]
    full       = [c for c in
                  scada_eng + list(cfg.device_features) + list(cfg.stress_features)
                  if c in train_df.columns]

    feature_sets = {
        "SCADA only":                      scada_eng,
        "SCADA + stress (deployment)":     scada_str,
        "SCADA + device physics":          scada_dev,
        "Full (SCADA + physics + stress)": full,
    }

    ablation_results = []
    deployment_model = None
    deployment_feats = None

    for set_name, feats in feature_sets.items():
        X_tr, used = build_feature_matrix(train_df, feats)
        X_te, _    = build_feature_matrix(test_df,  used)

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
