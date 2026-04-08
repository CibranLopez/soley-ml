"""
library.evaluation.temporal
=============================

Temporal split evaluation: train on years 1…N-1, test on year N.

run_temporal_split(registry, feature_names, output_dir, cfg)
    Reads sim_year from each file in the registry (train+val entries only),
    trains on all years except the last, tests on the last year.
    Files are streamed one at a time and subsampled, so peak RAM is bounded.
"""

import logging
import time
from pathlib import Path

import numpy as np

from library.features import build_feature_matrix, add_features

log = logging.getLogger("library")


def run_temporal_split(
    registry: list[dict],
    feature_names: list[str],
    output_dir: Path,
    cfg,
    max_rows_per_split: int = 400_000,
    random_state: int = 42,
) -> dict:
    """Train on early years, evaluate on the final year.

    Uses file-level entries from ``registry`` (train + val splits) so the
    same file boundary as all other models is respected.  Files are streamed
    one at a time; each split is capped at ``max_rows_per_split`` rows so
    the in-memory numpy matrices stay manageable.

    Parameters
    ----------
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` + ``assign_splits``.
    feature_names : list[str]
    output_dir : Path
    cfg : BatchConfig
    max_rows_per_split : int
        Maximum rows for train and test numpy arrays combined.
    random_state : int

    Returns
    -------
    dict  with keys ``detection_auc``, ``classification_auc`` (if available).
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    from library.visualization import plot_confusion

    log.info("")
    log.info("=" * 65)
    log.info("  TEMPORAL SPLIT (train years 1…N-1, test year N)")
    log.info("=" * 65)

    # Use train+val entries (test entries are reserved for held-out evaluation)
    entries = [e for e in registry if e.get("split") in ("train", "val")]
    if not entries:
        log.info("  No train/val entries in registry — skipping.")
        return {}

    rng = np.random.RandomState(random_state)

    # --- Step 1: discover sim_year per file by reading one column ----------
    log.info("  Discovering sim_year per file …")
    year_map: dict[str, int] = {}   # path → sim_year
    for e in entries:
        try:
            s = pd.read_parquet(e["path"], columns=["sim_year"]).iloc[0]["sim_year"]
            year_map[str(e["path"])] = int(s)
        except Exception:
            pass

    if not year_map:
        log.info("  sim_year column missing — skipping.")
        return {}

    years = sorted(set(year_map.values()))
    if len(years) < 2:
        log.info("  Only %d year(s) in train/val — skipping.", len(years))
        return {}

    test_year   = years[-1]
    train_years = set(years[:-1])
    log.info("  Train years: %s  |  Test year: %d",
             ", ".join(str(y) for y in sorted(train_years)), test_year)

    train_entries = [e for e in entries if year_map.get(str(e["path"])) in train_years]
    test_entries  = [e for e in entries if year_map.get(str(e["path"])) == test_year]
    log.info("  Files: train=%d  test=%d", len(train_entries), len(test_entries))

    # --- Step 2: stream & subsample each split ----------------------------
    def _stream(file_entries: list[dict], cap: int) -> pd.DataFrame:
        frames = []
        per_file = max(500, cap // max(len(file_entries), 1))
        for e in file_entries:
            df_e = pd.read_parquet(e["path"], columns=cfg.load_cols)
            df_e = add_features(df_e, cfg.array_kwp)
            if len(df_e) > per_file:
                df_e = df_e.sample(n=per_file, random_state=rng)
            if len(df_e):
                frames.append(df_e)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    log.info("  Streaming train files (cap=%s rows) …",
             f"{max_rows_per_split:,}")
    train_df = _stream(train_entries, max_rows_per_split)
    log.info("  Streaming test  files (cap=%s rows) …",
             f"{max_rows_per_split // 4:,}")
    test_df  = _stream(test_entries,  max_rows_per_split // 4)

    if train_df.empty or test_df.empty:
        log.info("  Not enough data after streaming — skipping.")
        return {}

    log.info("  Train: %s rows  Test: %s rows",
             f"{len(train_df):,}", f"{len(test_df):,}")

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
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    t0 = time.time()
    det_model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred  = det_model.predict(X_te)
    y_prob  = det_model.predict_proba(X_te)[:, 1]
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
        log.info("  Not enough fault types in test year — skipping cls.")
        return results

    all_faulted = pd.concat([train_f, test_f])
    le = LabelEncoder()
    le.fit(all_faulted["fault_type"].values)

    X_tr_f, _ = build_feature_matrix(train_f, feature_names)
    X_te_f, _ = build_feature_matrix(test_f,  feature_names)
    y_tr_f    = le.transform(train_f["fault_type"].values)
    y_te_f    = le.transform(test_f["fault_type"].values)

    cls_model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=10,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
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
