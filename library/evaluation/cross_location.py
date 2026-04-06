"""
library.evaluation.cross_location
=====================================

Leave-one-site-out cross-location generalisation test.

run_cross_location(df, feature_names, output_dir, cfg)
    For each unique site, train on all OTHER sites and test on that one.
    Returns a list of result dicts and saves a bar chart.
"""

import logging
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix

log = logging.getLogger("library")


def run_cross_location(
    df,
    feature_names: list[str],
    output_dir: Path,
    cfg,
) -> list[dict]:
    """Leave-one-site-out cross-location generalisation test.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with ``latitude`` and ``longitude`` columns.
    feature_names : list[str]
    output_dir : Path
    cfg : BatchConfig

    Returns
    -------
    list[dict]
        One dict per test site:
        ``{test_location, detection_auc, classification_auc, accuracy}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  CROSS-LOCATION GENERALISATION (leave-one-site-out)")
    log.info("=" * 65)

    df = df.copy()
    df["loc_id"] = df.apply(
        lambda r: cfg.get_loc_id(r["latitude"], r["longitude"]), axis=1,
    )
    locations = sorted(df["loc_id"].unique())

    if len(locations) < 2:
        log.info("  Need ≥ 2 locations — skipping.")
        return []

    all_faulted = df[df["fault_active"]]
    do_cls = (len(all_faulted) > 0
              and all_faulted["fault_type"].nunique() >= 2)
    if do_cls:
        le = LabelEncoder()
        le.fit(all_faulted["fault_type"].values)

    results = []

    for test_loc in locations:
        train_locs = [loc for loc in locations if loc != test_loc]
        train_df   = df[df["loc_id"].isin(train_locs)]
        test_df    = df[df["loc_id"] == test_loc]

        site_name = cfg.get_location_name(
            *[float(x) for x in test_loc.split("_")]
        )

        # Detection
        X_tr, _ = build_feature_matrix(train_df, feature_names)
        X_te, _ = build_feature_matrix(test_df,  feature_names)
        y_tr    = train_df["fault_active"].values.astype(int)
        y_te    = test_df["fault_active"].values.astype(int)

        det = RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        det.fit(X_tr, y_tr)

        try:
            auc_det = roc_auc_score(y_te, det.predict_proba(X_te)[:, 1])
        except ValueError:
            auc_det = float("nan")

        y_pred_det  = det.predict(X_te)
        acc_det     = (y_pred_det == y_te).mean()

        # Classification
        auc_cls = float("nan")
        if do_cls:
            train_f = train_df[train_df["fault_active"]]
            test_f  = test_df[test_df["fault_active"]]
            if test_f["fault_type"].nunique() >= 2:
                X_tr_f, _ = build_feature_matrix(train_f, feature_names)
                X_te_f, _ = build_feature_matrix(test_f,  feature_names)
                y_tr_f    = le.transform(train_f["fault_type"].values)
                y_te_f    = le.transform(test_f["fault_type"].values)

                cls = RandomForestClassifier(
                    n_estimators=100, max_depth=15,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )
                cls.fit(X_tr_f, y_tr_f)
                try:
                    auc_cls = roc_auc_score(
                        y_te_f, cls.predict_proba(X_te_f),
                        multi_class="ovr", average="weighted",
                    )
                except Exception:
                    pass

        log.info(
            "  Test: %-20s  Det AUC=%.4f  Cls AUC=%.4f  Acc=%.4f",
            site_name, auc_det, auc_cls, acc_det,
        )
        results.append({
            "test_location":    site_name,
            "test_loc_id":      test_loc,
            "detection_auc":    auc_det,
            "classification_auc": auc_cls,
            "accuracy":         acc_det,
        })

    # Plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names = [r["test_location"] for r in results]
        x     = range(len(results))

        for ax, key, color, title in [
            (axes[0], "detection_auc",      "#2563eb",
             "Cross-Location Detection\n(trained on OTHER sites)"),
            (axes[1], "classification_auc", "#10b981",
             "Cross-Location Classification\n(trained on OTHER sites)"),
        ]:
            vals = [r[key] for r in results]
            bars = ax.bar(x, vals, color=color, width=0.6)
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
            ax.set_ylabel("ROC AUC", fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.set_ylim(0, 1.08)
            ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random")
            ax.legend(fontsize=10)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                            f"{v:.3f}", ha="center", fontsize=11,
                            fontweight="bold")

        plt.tight_layout()
        out = output_dir / "cross_location.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  Saved cross_location.png")

    # Summary
    if results:
        avg_det = np.nanmean([r["detection_auc"]      for r in results])
        avg_cls = np.nanmean([r["classification_auc"] for r in results])
        log.info("  Average Detection AUC:       %.4f", avg_det)
        log.info("  Average Classification AUC:  %.4f", avg_cls)

    return results
