"""
library.evaluation.cross_location
=====================================

Leave-one-site-out cross-location generalisation test.

run_cross_location(registry, feature_names, output_dir, cfg)
    For each unique site, train on all OTHER sites and test on that one.
    Files are streamed one at a time and subsampled, keeping RAM bounded.
    Returns a list of result dicts and saves a bar chart.
"""

import logging
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix, add_features

log = logging.getLogger("library")


def run_cross_location(
    registry: list[dict],
    feature_names: list[str],
    output_dir: Path,
    cfg,
    max_rows_per_fold: int = 300_000,
    random_state: int = 42,
) -> list[dict]:
    """Leave-one-site-out cross-location generalisation test.

    Uses file-level entries from ``registry`` (train + val splits).
    Files are streamed and subsampled so each fold stays within
    ``max_rows_per_fold`` rows.

    Parameters
    ----------
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` + ``assign_splits``.
    feature_names : list[str]
    output_dir : Path
    cfg : BatchConfig
    max_rows_per_fold : int
        Row cap for the *training* split of each fold. Test uses ¼ of this.
    random_state : int

    Returns
    -------
    list[dict]
        One dict per test site:
        ``{test_location, detection_auc, classification_auc, accuracy}``.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    log.info("")
    log.info("=" * 65)
    log.info("  CROSS-LOCATION GENERALISATION (leave-one-site-out)")
    log.info("=" * 65)

    entries = [e for e in registry if e.get("split") in ("train", "val")]
    if not entries:
        log.info("  No train/val entries — skipping.")
        return []

    rng = np.random.RandomState(random_state)

    # --- Discover loc_id per file by reading lat/lon from first row -------
    log.info("  Discovering location per file …")
    loc_map: dict[str, str] = {}   # path → loc_id string
    for e in entries:
        try:
            row = pd.read_parquet(e["path"],
                                  columns=["latitude", "longitude"]).iloc[0]
            loc_id = cfg.get_loc_id(float(row["latitude"]),
                                    float(row["longitude"]))
            loc_map[str(e["path"])] = loc_id
        except Exception:
            pass

    if not loc_map:
        log.info("  latitude/longitude columns missing — skipping.")
        return []

    locations = sorted(set(loc_map.values()))
    if len(locations) < 2:
        log.info("  Need ≥ 2 locations (%d found) — skipping.",
                 len(locations))
        return []

    log.info("  Found %d locations: %s", len(locations),
             ", ".join(locations))

    # --- Helper: stream files for a set of loc_ids with row cap -----------
    def _stream(loc_ids: set, cap: int) -> pd.DataFrame:
        file_entries = [e for e in entries
                        if loc_map.get(str(e["path"])) in loc_ids]
        if not file_entries:
            return pd.DataFrame()
        per_file = max(200, cap // max(len(file_entries), 1))
        frames = []
        for e in file_entries:
            df_e = pd.read_parquet(e["path"], columns=cfg.load_cols)
            df_e = add_features(df_e, cfg.array_kwp)
            if len(df_e) > per_file:
                df_e = df_e.sample(n=per_file, random_state=rng)
            if len(df_e):
                frames.append(df_e)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Fit a single LabelEncoder over all files (for consistent class ids)
    log.info("  Fitting global LabelEncoder …")
    fault_types: list[str] = []
    for e in entries:
        try:
            col = pd.read_parquet(e["path"],
                                  columns=["fault_type", "fault_active"])
            fault_types.extend(
                col.loc[col["fault_active"], "fault_type"].dropna().unique()
            )
        except Exception:
            pass
    do_cls = len(set(fault_types)) >= 2
    if do_cls:
        le = LabelEncoder()
        le.fit(sorted(set(fault_types)))

    results = []

    for test_loc in locations:
        site_name  = cfg.get_location_name(
            *[float(x) for x in test_loc.split("_")]
        )
        train_locs = {loc for loc in locations if loc != test_loc}

        log.info("")
        log.info("  test_loc = %s (%s)", test_loc, site_name)
        train_df = _stream(train_locs,   max_rows_per_fold)
        test_df  = _stream({test_loc},   max_rows_per_fold // 4)

        if train_df.empty or test_df.empty:
            log.info("    Not enough data — skipping this fold.")
            continue

        log.info("    Train: %s rows  Test: %s rows",
                 f"{len(train_df):,}", f"{len(test_df):,}")

        # Detection
        X_tr, _ = build_feature_matrix(train_df, feature_names)
        X_te, _ = build_feature_matrix(test_df,  feature_names)
        y_tr    = train_df["fault_active"].values.astype(int)
        y_te    = test_df["fault_active"].values.astype(int)

        det = RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        det.fit(X_tr, y_tr)

        try:
            auc_det = roc_auc_score(y_te, det.predict_proba(X_te)[:, 1])
        except ValueError:
            auc_det = float("nan")

        y_pred_det = det.predict(X_te)
        acc_det    = (y_pred_det == y_te).mean()

        # Classification
        auc_cls = float("nan")
        if do_cls:
            train_f = train_df[train_df["fault_active"]]
            test_f  = test_df[test_df["fault_active"]]
            test_f_known = test_f[test_f["fault_type"].isin(le.classes_)]
            if test_f_known["fault_type"].nunique() >= 2:
                X_tr_f, _ = build_feature_matrix(train_f, feature_names)
                X_te_f, _ = build_feature_matrix(test_f_known, feature_names)
                # only include training fault types present in le
                mask = train_f["fault_type"].isin(le.classes_)
                X_tr_f, _ = build_feature_matrix(train_f[mask], feature_names)
                y_tr_f    = le.transform(train_f.loc[mask, "fault_type"].values)
                y_te_f    = le.transform(test_f_known["fault_type"].values)
                cls = RandomForestClassifier(
                    n_estimators=100, max_depth=15,
                    class_weight="balanced", random_state=random_state,
                    n_jobs=-1,
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
            "    Det AUC=%.4f  Cls AUC=%.4f  Acc=%.4f",
            auc_det, auc_cls, acc_det,
        )
        results.append({
            "test_location":      site_name,
            "test_loc_id":        test_loc,
            "detection_auc":      auc_det,
            "classification_auc": auc_cls,
            "accuracy":           acc_det,
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
        avg_det = np.nanmean([r["detection_auc"]       for r in results])
        avg_cls = np.nanmean([r["classification_auc"]  for r in results])
        log.info("  Average Detection AUC:       %.4f", avg_det)
        log.info("  Average Classification AUC:  %.4f", avg_cls)

    return results
