"""
library.models.dataset
========================

Memory-safe PyTorch dataset for temporal-window fault detection.

WindowDataset
    PyTorch ``Dataset`` backed by disk-mapped numpy arrays (memmap).
    Supports windows of any size and stride.  Opens memmap handles
    lazily with an LRU cache to stay within OS file-descriptor limits.

build_memmap_arrays(entries, feature_cols, target_col, cfg, scaler, le)
    Streams parquet files one at a time, applies feature engineering,
    scales features, and writes the result to disk-backed memmap files.
    Peak RAM ≈ 1 parquet file (~60–100 MB), regardless of dataset size.

fit_scaler_streaming(entries, feature_cols, cfg, n_sample_files)
    Fits a StandardScaler on a random sample of training files.
"""

import bisect
import gc
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import BatchConfig

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Scaler fitting
# ---------------------------------------------------------------------------

def fit_scaler_streaming(
    entries: list[dict],
    feature_cols: list[str],
    cfg: BatchConfig,
    n_sample_files: int = 20,
    lookback_window: int = 12,
):
    """Fit a StandardScaler by sampling rows from up to ``n_sample_files`` files.

    Parameters
    ----------
    entries : list[dict]
        Registry entries (each has a ``"path"`` key).
    feature_cols : list[str]
    cfg : BatchConfig
    n_sample_files : int
    lookback_window : int
        Passed to :func:`~library.features.add_features`.  Must match
        ``window_size`` used in :class:`WindowDataset` so that feature
        engineering is consistent between the scaler-fitting step and
        inference.  Default: ``12``.

    Returns
    -------
    scaler : StandardScaler
    available : list[str]
        Feature columns that were actually found in the files.
    """
    from sklearn.preprocessing import StandardScaler

    from library.data.loader import load_registry_entry  # single canonical reader

    step    = max(1, len(entries) // n_sample_files)
    sampled = entries[::step][:n_sample_files]

    chunks    = []
    available = None

    for e in sampled:
        df = load_registry_entry(e, cfg, lookback_window=lookback_window)
        if available is None:
            available = [c for c in feature_cols if c in df.columns]
        n = min(10_000, len(df))
        if n > 0:
            chunks.append(df[available].sample(n=n, random_state=42))
        del df

    combined = (pd.concat(chunks)
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0))
    scaler = StandardScaler()
    scaler.fit(combined.values.astype(np.float32))
    del chunks, combined
    gc.collect()

    log.info("  Scaler fit on %d files, %d features",
             len(sampled), len(available))
    return scaler, available


# ---------------------------------------------------------------------------
#  Memmap builder
# ---------------------------------------------------------------------------

def build_memmap_arrays(
    entries: list[dict],
    feature_cols: list[str],
    target_col: str,
    cfg: BatchConfig,
    scaler,
    label_encoder=None,
    faulted_only: bool = False,
    lookback_window: int = 12,
) -> tuple[list[tuple], Path]:
    """Stream parquet files to disk-backed memmap arrays.

    Parameters
    ----------
    entries : list[dict]
        Registry entries with ``"path"`` key.
    feature_cols : list[str]
        The ``available`` list returned by :func:`fit_scaler_streaming`.
    target_col : str
        ``"fault_active"`` (detection) or ``"fault_type"`` (classification).
    cfg : BatchConfig
    scaler : fitted StandardScaler
    label_encoder : LabelEncoder or None
        Required when ``target_col == "fault_type"``.
    faulted_only : bool
        If True, only keep faulted rows (for classification).
    lookback_window : int
        Passed to :func:`~library.features.add_features`.  Must match
        ``window_size`` used in :class:`WindowDataset` so that the rolling
        features computed here are reproducible at inference time.
        Default: ``12``.

    Returns
    -------
    runs : list[tuple]
        ``(feat_path, feat_shape, lab_path, lab_shape, run_id)``
    cache_dir : Path
        Temporary directory containing the memmap files.
        Call ``shutil.rmtree(cache_dir)`` when done.
    """
    cache_dir = Path(tempfile.mkdtemp(prefix="library_"))
    runs      = []

    from library.data.loader import load_registry_entry  # single canonical reader

    for i, e in enumerate(entries):
        df = load_registry_entry(e, cfg, lookback_window=lookback_window)

        if faulted_only and "fault_active" in df.columns:
            df = df[df["fault_active"]].copy()
        if target_col != "fault_active" and label_encoder is not None and len(df):
            known = df["fault_type"].isin(label_encoder.classes_)
            dropped = int((~known).sum())
            if dropped:
                log.info("    Skipping %d rows with unseen fault types in %s",
                         dropped, e["path"].name)
            df = df.loc[known].copy()
        if len(df) == 0:
            del df
            continue

        df = df.sort_values("timestamp")

        # Feature matrix
        feat = df[feature_cols].values.astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = scaler.transform(feat).astype(np.float32)

        # Labels
        if target_col == "fault_active":
            labels = df["fault_active"].astype(int).values.astype(np.int64)
        else:
            labels = label_encoder.transform(
                df["fault_type"].values
            ).astype(np.int64)

        # Write and immediately close (avoids fd accumulation)
        fp = cache_dir / f"feat_{i}.dat"
        lp = cache_dir / f"lab_{i}.dat"

        fm = np.memmap(fp, dtype=np.float32, mode="w+", shape=feat.shape)
        fm[:] = feat
        fm.flush()
        feat_shape = feat.shape
        del fm

        lm = np.memmap(lp, dtype=np.int64, mode="w+", shape=labels.shape)
        lm[:] = labels
        lm.flush()
        lab_shape = labels.shape
        del lm

        runs.append((str(fp), feat_shape, str(lp), lab_shape, e["run_id"]))

        del df, feat, labels
        if (i + 1) % 50 == 0:
            gc.collect()
            log.info("    … %d / %d files", i + 1, len(entries))

    gc.collect()
    log.info("  Cached %d runs to disk", len(runs))
    return runs, cache_dir


# ---------------------------------------------------------------------------
#  WindowDataset
# ---------------------------------------------------------------------------

class WindowDataset:
    """PyTorch Dataset: temporal windows from disk-backed numpy arrays.

    Uses a compact index (one entry per run rather than one per window)
    to minimise Python object overhead for large datasets.

    Memmap files are opened lazily and cached with an LRU policy
    to respect OS file-descriptor limits.

    .. note::
        The ``window_size`` used here **must match** the ``lookback_window``
        passed to :func:`~library.features.add_features` (via
        :func:`~library.data.loader.load_data_pt`,
        :func:`~library.models.dataset.fit_scaler_streaming`, and
        :func:`~library.models.dataset.build_memmap_arrays`).  Both default
        to ``12`` so that the Random Forest and sequence models see identical
        rolling-feature context out of the box.

    Parameters
    ----------
    run_arrays : list[tuple]
        ``(feat_path, feat_shape, lab_path, lab_shape, run_id)``
        as returned by :func:`build_memmap_arrays`.
    window_size : int
        Number of consecutive timesteps per sample.
    stride : int
        Step between windows.
    cache_max : int
        Maximum simultaneously open memmap file-handle pairs.
    """

    def __init__(self, run_arrays: list[tuple],
                 window_size: int, stride: int,
                 cache_max: int = 64):
        self.run_arrays  = run_arrays
        self.window_size = window_size
        self.stride      = stride

        self._run_info   = []   # (run_idx, n_windows)
        self._cumulative = []   # cumulative window count before this run
        self._total      = 0

        for run_idx, (fp, fs, lp, ls, rid) in enumerate(run_arrays):
            n = fs[0]
            if n < window_size:
                continue
            n_windows = (n - window_size) // stride + 1
            self._run_info.append((run_idx, n_windows))
            self._cumulative.append(self._total)
            self._total += n_windows

        self._cache       = {}
        self._cache_order = []
        self._cache_max   = cache_max

    def _get_arrays(self, run_idx: int):
        if run_idx in self._cache:
            return self._cache[run_idx]

        fp, fs, lp, ls, _ = self.run_arrays[run_idx]
        feat = np.memmap(fp, dtype=np.float32, mode="r", shape=fs)
        lab  = np.memmap(lp, dtype=np.int64,   mode="r", shape=ls)

        if len(self._cache_order) >= self._cache_max:
            evict_idx = self._cache_order.pop(0)
            old = self._cache.pop(evict_idx, None)
            if old is not None:
                del old

        self._cache[run_idx] = (feat, lab)
        self._cache_order.append(run_idx)
        return feat, lab

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        import torch
        pos      = bisect.bisect_right(self._cumulative, idx) - 1
        run_idx, _ = self._run_info[pos]
        local_idx  = idx - self._cumulative[pos]
        start      = local_idx * self.stride

        feat, labels = self._get_arrays(run_idx)
        end    = start + self.window_size
        window = torch.from_numpy(feat[start:end].copy())
        label  = torch.tensor(int(labels[end - 1]), dtype=torch.long)
        return window, label
