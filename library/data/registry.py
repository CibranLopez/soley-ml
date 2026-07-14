"""
library.data.registry
=======================

File-level metadata registry used by the PyTorch streaming pipeline.

prepare_file_registry(data_dir, cfg, max_runs, location_filter)
    Discovers parquet files, extracts fault types from filenames,
    and optionally limits to ``max_runs`` balanced across fault types.
    **No data is loaded** — only paths and metadata are returned.

assign_splits(registry, seed)
    Stratified 60/20/20 train/val/test split by fault type.
    Each entry in ``registry`` receives a ``"split"`` key.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from library.config import BatchConfig

log = logging.getLogger("library")


def _fault_type_from_filename(filepath: Path) -> str:
    name = filepath.stem
    for suffix in ("_noisy", "_clean"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if "_fault_" in name:
        after = name.split("_fault_")[1]
        for sep in ("_af", "_aging"):
            if sep in after:
                return after.split(sep)[0]
        return after
    if "_combo" in name:
        return "combo"
    return "none"


def prepare_file_registry(
    data_dir: str | Path,
    cfg: BatchConfig,
    max_runs: int | None = None,
    location_filter: str | None = None,
) -> list[dict]:
    """Discover parquet files and extract metadata.

    No data is loaded at this stage.

    Parameters
    ----------
    data_dir : str or Path
    cfg : BatchConfig
    max_runs : int or None
        Limit to this many files, balanced across fault types.
    location_filter : str or None
        Only include files whose name contains this string.

    Returns
    -------
    list[dict]
        Each entry has keys ``path``, ``run_id``, ``fault_type``.
        After calling :func:`assign_splits`, each entry also has ``split``.
    """
    data_dir = Path(data_dir)
    files    = sorted(data_dir.glob("*_noisy.parquet"))
    if not files:
        files = sorted(data_dir.glob("*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    if location_filter:
        files = [f for f in files if location_filter in f.name]

    registry = [
        {
            "path":       f,
            "run_id":     f.stem.split("_")[0],
            "fault_type": _fault_type_from_filename(f),
        }
        for f in files
    ]

    # Limit runs: balanced across fault types
    if max_runs and len(registry) > max_runs:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in registry:
            groups[e["fault_type"]].append(e)
        per_type = max(1, max_runs // len(groups))
        selected = []
        for ft in sorted(groups):
            selected.extend(groups[ft][:per_type])
        registry = sorted(selected, key=lambda e: str(e["path"]))[:max_runs]

    fault_types = sorted({e["fault_type"] for e in registry})
    log.info("Registry: %d files, %d fault types", len(registry), len(fault_types))
    return registry


def assign_splits(registry: list[dict], val_size: float = 0.20, test_size: float = 0.20, seed: int = 42) -> None:
    """Stratified 60 / 20 / 20 train / val / test split.

    Modifies ``registry`` **in place** — each entry receives a
    ``"split"`` key (``"train"``, ``"val"``, or ``"test"``).

    Parameters
    ----------
    registry : list[dict]
        From :func:`prepare_file_registry`.
    val_size : float
        Proportion of data to allocate to the validation set.
    test_size : float
        Proportion of data to allocate to the test set.
    seed : int
    """
    rng    = np.random.RandomState(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in registry:
        groups[e["fault_type"]].append(e)

    for ft, entries in groups.items():
        rng.shuffle(entries)
        n = len(entries)
        if n >= 3:
            n_val   = max(1, int(val_size * n))
            n_test  = max(1, int(test_size * n))
            n_train = n - n_val - n_test
            for e in entries[:n_train]:
                e["split"] = "train"
            for e in entries[n_train: n_train + n_val]:
                e["split"] = "val"
            for e in entries[n_train + n_val:]:
                e["split"] = "test"
        elif n == 2:
            entries[0]["split"] = "train"
            entries[1]["split"] = "test"
        else:
            entries[0]["split"] = "train"

    for split in ("train", "val", "test"):
        count = sum(1 for e in registry if e.get("split") == split)
        log.info("  %s: %d files", split, count)


# ---------------------------------------------------------------------------
#  Manifest persistence (registry + split assignment + saved model paths)
# ---------------------------------------------------------------------------
#
# Why a manifest, not "split files" or row-level masks
# ------------------------------------------------------
# The registry is already file-level metadata (one dict per parquet file,
# not per row), so the natural unit to persist is exactly that: a small
# JSON list of {path, run_id, fault_type, split} records. This is *not*
# the same as physically copying parquet files into train/ val/ test/
# directories (which would duplicate potentially many GB of data and
# create a second source of truth that can drift from the original files)
# and it doesn't need row-level boolean masks (those make sense when you
# split individual rows of one in-memory array; here the split unit is a
# whole file, so a list of per-file split labels *is* the mask, just at
# file granularity instead of row granularity). Persisting the manifest
# is simply serializing the exact `registry` list that
# `prepare_file_registry` + `assign_splits` already build in memory, with
# `Path` objects converted to ``str`` for JSON and restored to `Path` on
# load so every downstream consumer (loader functions, evaluation
# studies, etc.) sees the same shape it already expects.

def save_registry_manifest(
    registry: list[dict],
    manifest_path: str | Path,
    *,
    model_paths: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist a file-level registry (with assigned splits) to a JSON manifest.

    This is the structural fix for the train/eval split-leakage risk: instead
    of notebook 3 re-deriving the registry and re-running
    :func:`assign_splits` from scratch (which silently diverges from
    training's registry whenever ``max_runs`` differs, even with the same
    seed — same seed, different input list), notebook 2 calls this once at
    the end of training and notebook 3 loads the result directly via
    :func:`load_registry_manifest`. There is then exactly one place where
    train/val/test membership is decided.

    Also doubles as the fix for the RF/PT artifact-path mismatch: pass the
    ``model_path`` values already returned by
    ``run_model_task(..., return_details=True)`` (for every mode — RF and
    PyTorch alike) as ``model_paths`` and notebook 3 can look paths up
    directly instead of reconstructing them from hardcoded directory /
    filename conventions.

    Parameters
    ----------
    registry : list[dict]
        From :func:`prepare_file_registry` + :func:`assign_splits`. Each
        entry's ``"path"`` (a ``Path``) is serialized as ``str``.
    manifest_path : str or Path
        Output JSON file. Parent directories are created if needed.
    model_paths : dict or None
        Arbitrary nested mapping of saved model artifact paths, e.g.
        ``{"fault_active": {"random_forest": "...", "mlp": "...", ...},
        "fault_type": {...}}``. Stored as-is and returned unchanged by
        :func:`load_registry_manifest`.
    extra : dict or None
        Any additional metadata to stash alongside the manifest (e.g. the
        seed used, data_dir, max_runs at training time) purely for
        provenance / debugging — not required by any loader.

    Returns
    -------
    Path
        ``manifest_path``, for convenient chaining.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_entries = []
    for e in registry:
        entry = dict(e)
        entry["path"] = str(entry["path"])
        serializable_entries.append(entry)

    manifest = {
        "version": 1,
        "n_entries": len(serializable_entries),
        "registry": serializable_entries,
        "model_paths": model_paths or {},
    }
    if extra:
        manifest["extra"] = extra

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("Saved registry manifest: %s  (%d files, %d split-test)",
             manifest_path, len(serializable_entries),
             sum(1 for e in serializable_entries if e.get("split") == "test"))
    return manifest_path


def load_registry_manifest(manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Load a registry manifest written by :func:`save_registry_manifest`.

    Parameters
    ----------
    manifest_path : str or Path

    Returns
    -------
    registry : list[dict]
        Each entry's ``"path"`` is restored to a ``Path`` object so it is a
        drop-in replacement for the output of
        ``prepare_file_registry`` + ``assign_splits`` — no recomputation,
        no re-running of either function.
    model_paths : dict
        Whatever was passed as ``model_paths`` to
        :func:`save_registry_manifest` (``{}`` if none was saved).
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    registry = []
    for e in manifest["registry"]:
        entry = dict(e)
        entry["path"] = Path(entry["path"])
        registry.append(entry)

    model_paths = manifest.get("model_paths", {})

    split_counts = defaultdict(int)
    for e in registry:
        split_counts[e.get("split", "?")] += 1
    log.info("Loaded registry manifest: %s  (%d files: %s)",
             manifest_path, len(registry),
             ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))
    return registry, model_paths
