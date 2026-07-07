"""
library.evaluation.unified
==========================

Model-agnostic evaluation helpers built on top of the unified training API.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from library.data import assign_splits
from library.models import run_model_task
from library.visualization import plot_ablation, plot_cross_location

log = logging.getLogger("library")


def _task_slug(task_name: str) -> str:
    return task_name.lower().replace(" ", "_")


def _clone_with_splits(entries: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    pool = [dict(e) for e in entries]
    assign_splits(pool, seed=seed)
    train_entries = [e for e in pool if e.get("split") == "train"]
    val_entries = [e for e in pool if e.get("split") == "val"]
    return train_entries, val_entries


def _run_dual_tasks(
    *,
    registry: list[dict],
    feature_names: list[str],
    output_dir: Path,
    cfg,
    model_modes: list[str],
    train_entries: list[dict],
    val_entries: list[dict],
    test_entries: list[dict],
    window_size: int | None,
    stride: int | None,
    batch_size: int | None,
    epochs: int | None,
    device,
    patience: int | None,
    num_workers: int,
    rf_params: dict | None,
    custom_runners: dict[str, callable] | None,
    prefix: str,
) -> dict:
    tasks = [
        ("Fault Detection", "fault_active", "detection_auc"),
        ("Fault Classification", "fault_type", "classification_auc"),
    ]
    aggregated: dict[str, dict] = {}

    for task_name, target_col, metric_key in tasks:
        details = run_model_task(
            task_name=task_name,
            target_col=target_col,
            registry=registry,
            feature_cols=feature_names,
            model_modes=model_modes,
            output_dir=output_dir,
            cfg=cfg,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            patience=patience,
            num_workers=num_workers,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            artifact_prefix=f"{prefix}_{_task_slug(task_name)}",
            rf_params=rf_params,
            custom_runners=custom_runners,
            return_details=True,
        )
        for mode, meta in details.items():
            aggregated.setdefault(mode, {})[metric_key] = meta.get("auc")

    return aggregated


def run_temporal_split(
    registry: list[dict],
    feature_names: list[str],
    output_dir: Path,
    cfg,
    *,
    model_modes: list[str] | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    device=None,
    patience: int | None = None,
    num_workers: int = 0,
    rf_params: dict | None = None,
    custom_runners: dict[str, callable] | None = None,
    random_state: int = 42,
) -> dict:
    """Train on early sim years and evaluate on the final sim year.

    Temporal split design for multi-year parquet files
    ---------------------------------------------------
    Each SOLEY parquet file can span multiple sim_year values (e.g. years
    1–6 representing 2018–2023 in a single run file).  Reading only the
    first row to discover a file's year — as the naive implementation did —
    always returns year 1 for every file, collapsing all files into one
    year bucket and making the split impossible.

    The fix: read all unique ``sim_year`` values from every file to build
    a global year set, then build year-filtered entry dicts
    (``{"path": ..., "sim_year_filter": [1,2,3,4]}`` for training,
    ``{"path": ..., "sim_year_filter": [6]}`` for testing).  The tabular
    loader (``_load_train_test_from_registry``) and the PyTorch loader
    both recognise the ``sim_year_filter`` key and filter rows accordingly,
    so no data is physically duplicated and no new files need to be created.
    """
    model_modes = model_modes or ["random_forest"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in registry if e.get("split") in ("train", "val")]
    if not entries:
        return {}

    log.info("")
    log.info("=" * 65)
    log.info("  TEMPORAL SPLIT (all model modes)")
    log.info("=" * 65)

    # --- Discover ALL sim_years per file, not just row 0 ------------------
    # Row 0 is always the earliest timestamp (year 1) in a multi-year file,
    # so reading .iloc[0] would assign every file to year 1 and make the
    # split degenerate.  We read the full sim_year column (cheap — one int
    # column) and collect unique values instead.
    log.info("  Discovering sim_year range per file …")
    file_years: dict[str, set[int]] = {}   # path → set of years in that file
    for e in entries:
        try:
            ys = pd.read_parquet(e["path"], columns=["sim_year"])["sim_year"].unique()
            file_years[str(e["path"])] = {int(y) for y in ys}
        except Exception:
            pass

    if not file_years:
        log.info("  sim_year column missing in registry files — skipping.")
        return {}

    all_years = sorted(set.union(*file_years.values()))
    log.info("  sim_years found across all files: %s", all_years)

    if len(all_years) < 2:
        log.info("  Only %d sim_year value(s) found — need ≥ 2 for a "
                 "temporal split — skipping.", len(all_years))
        return {}

    test_year   = all_years[-1]
    train_years = set(all_years[:-1])
    log.info("  Train years: %s  |  Test year: %d",
             sorted(train_years), test_year)

    # --- Build year-filtered entries --------------------------------------
    # Each entry dict gets a "sim_year_filter" key; the tabular loader
    # applies it as a row-level filter after reading the parquet file.
    # This means every file contributes rows to BOTH the train and test
    # split (filtered to different year ranges), which is the correct
    # behaviour when files span multiple years.
    train_pool    = [dict(e, sim_year_filter=sorted(train_years)) for e in entries]
    test_entries  = [dict(e, sim_year_filter=[test_year])         for e in entries]

    # Internal train/val split for early-stopping in sequence models
    train_entries, val_entries = _clone_with_splits(train_pool, seed=random_state)

    log.info("  train_entries=%d  val_entries=%d  test_entries=%d",
             len(train_entries), len(val_entries), len(test_entries))

    results = _run_dual_tasks(
        registry=registry,
        feature_names=feature_names,
        output_dir=output_dir,
        cfg=cfg,
        model_modes=model_modes,
        train_entries=train_entries,
        val_entries=val_entries,
        test_entries=test_entries,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        patience=patience,
        num_workers=num_workers,
        rf_params=rf_params,
        custom_runners=custom_runners,
        prefix="temporal",
    )

    for mode, metrics in results.items():
        log.info("  %-16s detection_auc=%s classification_auc=%s",
                 mode,
                 f"{metrics.get('detection_auc'):.4f}" if metrics.get("detection_auc") is not None else "n/a",
                 f"{metrics.get('classification_auc'):.4f}" if metrics.get("classification_auc") is not None else "n/a")
    return results


def run_cross_location(
    registry: list[dict],
    feature_names: list[str],
    output_dir: Path,
    cfg,
    *,
    model_modes: list[str] | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    device=None,
    patience: int | None = None,
    num_workers: int = 0,
    rf_params: dict | None = None,
    custom_runners: dict[str, callable] | None = None,
    random_state: int = 42,
) -> list[dict]:
    """Leave-one-site-out evaluation for each selected model mode."""
    model_modes = model_modes or ["random_forest"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in registry if e.get("split") in ("train", "val")]
    if not entries:
        return []

    loc_map: dict[str, str] = {}
    for e in entries:
        try:
            row = pd.read_parquet(e["path"], columns=["latitude", "longitude"]).iloc[0]
            loc_map[str(e["path"])] = cfg.get_loc_id(float(row["latitude"]), float(row["longitude"]))
        except Exception:
            pass

    locations = sorted(set(loc_map.values()))
    if len(locations) < 2:
        log.info("  Need at least two locations for cross-location evaluation.")
        return []

    rows = []
    for test_loc in locations:
        train_pool = [e for e in entries if loc_map.get(str(e["path"])) != test_loc]
        test_entries = [e for e in entries if loc_map.get(str(e["path"])) == test_loc]
        train_entries, val_entries = _clone_with_splits(train_pool, seed=random_state)
        metrics_by_mode = _run_dual_tasks(
            registry=registry,
            feature_names=feature_names,
            output_dir=output_dir,
            cfg=cfg,
            model_modes=model_modes,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            patience=patience,
            num_workers=num_workers,
            rf_params=rf_params,
            custom_runners=custom_runners,
            prefix=f"cross_location_{test_loc}",
        )
        site_name = cfg.get_location_name(*[float(x) for x in test_loc.split("_")])
        for mode, metrics in metrics_by_mode.items():
            # metrics.get(key, np.nan) is NOT sufficient here: it only
            # substitutes np.nan when the key is entirely absent, not when
            # the key is present but its value is an explicit None — which
            # is exactly what classification_metrics() stores when AUC
            # computation fails (e.g. a fold whose test data is degenerate
            # for classification). An explicit None previously slipped
            # through into plot_cross_location's ax.bar() call, which
            # crashed (matplotlib can plot NaN as a gap, but not None).
            det_auc = metrics.get("detection_auc")
            cls_auc = metrics.get("classification_auc")
            rows.append({
                "model_mode": mode,
                "test_location": site_name,
                "test_loc_id": test_loc,
                "detection_auc": det_auc if det_auc is not None else np.nan,
                "classification_auc": cls_auc if cls_auc is not None else np.nan,
            })

    for mode in sorted({row["model_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["model_mode"] == mode]
        plot_cross_location(mode_rows, output_dir / f"cross_location_{mode}.png")

    return rows


def run_feature_ablation(
    registry: list[dict],
    output_dir: Path,
    cfg,
    *,
    model_modes: list[str] | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    device=None,
    patience: int | None = None,
    num_workers: int = 0,
    rf_params: dict | None = None,
    custom_runners: dict[str, callable] | None = None,
) -> list[dict]:
    """Detection ablation study across feature sets for each selected mode."""
    model_modes = model_modes or ["random_forest"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_entries = [e for e in registry if e.get("split") == "train"]
    val_entries = [e for e in registry if e.get("split") == "val"]
    test_entries = [e for e in registry if e.get("split") == "test"]

    base_cols = cfg.feature_set("full")
    feature_sets = {
        "SCADA only": cfg.feature_set("scada"),
        "SCADA + stress": [c for c in base_cols if c in set(cfg.scada_features) | set(cfg.stress_features)],
        "SCADA + device physics": [c for c in base_cols if c in set(cfg.scada_features) | set(cfg.device_features)],
        "Full": base_cols,
    }

    rows = []
    for set_name, feats in feature_sets.items():
        details = run_model_task(
            task_name="Fault Detection",
            target_col="fault_active",
            registry=registry,
            feature_cols=feats,
            model_modes=model_modes,
            output_dir=output_dir,
            cfg=cfg,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            patience=patience,
            num_workers=num_workers,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            artifact_prefix=f"ablation_{set_name.lower().replace(' ', '_').replace('+', 'plus')}",
            rf_params=rf_params,
            custom_runners=custom_runners,
            return_details=True,
        )
        for mode, meta in details.items():
            # meta.get("auc") can be an explicit None (classification_metrics
            # stores None when AUC computation itself fails, e.g. a
            # degenerate test fold with only one class present) — substitute
            # np.nan so this can't reach plot_ablation's min(aucs) call
            # un-guarded, matching the fix applied to run_cross_location for
            # the identical root cause.
            auc_val = meta.get("auc")
            rows.append({
                "model_mode": mode,
                "feature_set": set_name,
                "n_features": len(feats),
                "auc": auc_val if auc_val is not None else np.nan,
            })

    for mode in sorted({row["model_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["model_mode"] == mode]
        plot_ablation(mode_rows, output_dir / f"feature_ablation_{mode}.png")

    return rows


__all__ = ["run_temporal_split", "run_cross_location", "run_feature_ablation"]