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
    """Train on early years and evaluate on the last year for each model mode."""
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

    year_map: dict[str, int] = {}
    for e in entries:
        try:
            sim_year = pd.read_parquet(e["path"], columns=["sim_year"]).iloc[0]["sim_year"]
            year_map[str(e["path"])] = int(sim_year)
        except Exception:
            pass

    years = sorted(set(year_map.values()))
    if len(years) < 2:
        log.info("  Not enough years for temporal evaluation.")
        return {}

    test_year = years[-1]
    train_pool = [e for e in entries if year_map.get(str(e["path"])) in set(years[:-1])]
    test_entries = [e for e in entries if year_map.get(str(e["path"])) == test_year]
    train_entries, val_entries = _clone_with_splits(train_pool, seed=random_state)

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
            rows.append({
                "model_mode": mode,
                "test_location": site_name,
                "test_loc_id": test_loc,
                "detection_auc": metrics.get("detection_auc", np.nan),
                "classification_auc": metrics.get("classification_auc", np.nan),
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
            rows.append({
                "model_mode": mode,
                "feature_set": set_name,
                "n_features": len(feats),
                "auc": meta.get("auc"),
            })

    for mode in sorted({row["model_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["model_mode"] == mode]
        plot_ablation(mode_rows, output_dir / f"feature_ablation_{mode}.png")

    return rows


__all__ = ["run_temporal_split", "run_cross_location", "run_feature_ablation"]