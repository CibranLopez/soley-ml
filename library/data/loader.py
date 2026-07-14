"""
library.data.loader
====================

Two-function loading interface — the only permitted way to read SOLEY
parquet data in this library:

load_registry_entry(entry, cfg, lookback_window)
    **Atomic unit.** Reads one parquet file identified by a registry entry,
    applies an optional sim_year_filter, and runs the standard feature-
    engineering pipeline (add_features).  All other loading functions in
    the library are wrappers around this one.

load_split(entries, cfg, max_rows, lookback_window, random_state)
    **Multi-entry collector.** Iterates over a list of registry entries,
    calls load_registry_entry for each, concatenates the results, and
    optionally subsamples the concatenated DataFrame in a stratified manner.
    Used by the training pipeline (tabular models) and by exploration
    notebooks.

Why only two functions?
-----------------------
The previous load_data_rf / load_data_pt functions read files by scanning a
directory, had their own sampling strategies, and were completely decoupled
from the registry split assignments.  Using them alongside the registry-based
training pipeline created four separate loading paths with different year-
filter, lookback_window, and sampling behaviour — making cross-model
comparisons meaningless and temporal-split evaluation silently wrong for RF.

Having a single atom (load_registry_entry) that every model family routes
through guarantees:
* Identical column selection (cfg.load_cols).
* Identical row-level year filter (entry["sim_year_filter"]) — required for
  run_temporal_split to work for every model, not just the PyTorch one.
* Identical feature engineering (add_features with the same lookback_window)
  so rolling PR-deviation and power-step are bit-for-bit equal across RF,
  MLP, LSTM, and Hybrid.
* No leakage: the registry split (assigned once by assign_splits) is the
  only authority on which files are train / val / test.
"""

import logging

import numpy as np
import pandas as pd

from library.config import BatchConfig
from library.features import add_features

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Shared read-with-year-filter primitive
# ---------------------------------------------------------------------------

def _read_parquet_with_year_filter(
    path,
    columns: list[str],
    sim_year_filter: list[int] | None,
):
    """Read a parquet file and apply an optional row-level ``sim_year`` filter.

    Factored out of :func:`load_registry_entry` and :func:`discover_fault_types`,
    which both need the identical "add sim_year to the read if filtering,
    then filter rows" logic but over different column lists (the full
    feature set vs. two label columns). Keeping this in one place means
    the year-filter behaviour — used by
    :func:`~library.evaluation.unified.run_temporal_split` — can't
    silently diverge between the two callers.

    Parameters
    ----------
    path : Path or str
    columns : list[str]
        Columns to read, before any ``sim_year`` addition.
    sim_year_filter : list[int] or None
        If given, ``sim_year`` is added to the read (if not already
        present) and rows outside this set of years are dropped.

    Returns
    -------
    pd.DataFrame
    """
    cols = list(columns)
    if sim_year_filter and "sim_year" not in cols:
        cols = cols + ["sim_year"]

    df = pd.read_parquet(path, columns=cols)

    if sim_year_filter:
        df = df[df["sim_year"].isin(sim_year_filter)]

    return df


# ---------------------------------------------------------------------------
#  Atom: one registry entry → one DataFrame
# ---------------------------------------------------------------------------

def load_registry_entry(
    entry: dict,
    cfg: BatchConfig,
    lookback_window: int = 12,
) -> pd.DataFrame:
    """Load one registry entry's parquet file, apply optional year filter,
    and run the standard feature-engineering pipeline.

    This is the **single canonical data-reading function** for the entire
    library.  Every model family — Random Forest, MLP, LSTM, Hybrid, and
    any future tabular estimator — must obtain its data through this
    function so that preprocessing is provably identical across models.

    Parameters
    ----------
    entry : dict
        Registry entry with at minimum ``"path"`` (``Path`` or ``str``).
        Optionally ``"sim_year_filter": list[int]`` — added automatically
        by :func:`~library.evaluation.unified.run_temporal_split` when
        building year-partitioned train / test entry lists from multi-year
        parquet files.
    cfg : BatchConfig
        Provides ``cfg.load_cols`` (column list) and ``cfg.array_kwp``
        (for performance-ratio computation in add_features).
    lookback_window : int
        Forwarded to :func:`~library.features.add_features` as the rolling-
        window size for PR-deviation and power-step computation.  Must be
        the same value across all callers in one training run — enforced by
        threading it from ``run_model_task`` down through both backends.
        Default ``12`` ≈ 1 h at 5-min resolution.

    Returns
    -------
    pd.DataFrame
        Daytime-filtered (poa_global_wm2 > 10 W/m²), feature-engineered
        slice of the parquet file.  May be empty after daytime or year
        filtering — callers must check ``len(df) > 0`` before use.
    """
    df = _read_parquet_with_year_filter(
        entry["path"], cfg.load_cols, entry.get("sim_year_filter"))
    return add_features(df, cfg.array_kwp, lookback_window=lookback_window)


# ---------------------------------------------------------------------------
#  Cheap label discovery — real data, not filename metadata
# ---------------------------------------------------------------------------

def discover_fault_types(entries: list[dict]) -> list[str]:
    """Discover every ``fault_type`` value that actually appears in the
    faulted (``fault_active=True``) rows of the given registry entries.

    Reads only the ``fault_type`` / ``fault_active`` / ``sim_year``
    columns — skips every feature column and the
    :func:`~library.features.add_features` pipeline entirely — so this is
    cheap to call even over hundreds of files.

    Used by :func:`~library.models.trainer.run_task` to fit a single
    shared ``LabelEncoder`` for every model family from one source of
    truth: real row-level data, not filename-derived metadata. A file's
    name only encodes its primary fault type (used for stratifying the
    train/val/test split); reading the actual rows is the only way to be
    certain what ``fault_type`` values a file's data really contains.

    Parameters
    ----------
    entries : list[dict]
        Registry entries with ``"path"`` and optionally
        ``"sim_year_filter"`` (respected the same way as
        :func:`load_registry_entry`).

    Returns
    -------
    list[str]
        Sorted, deduplicated ``fault_type`` values across every given
        entry's faulted rows. Empty list if none of the entries have any
        faulted rows (e.g. an empty entry list, or every row filtered out
        by ``sim_year_filter``).
    """
    types: set[str] = set()
    for e in entries:
        try:
            df = _read_parquet_with_year_filter(
                e["path"], ["fault_type", "fault_active"], e.get("sim_year_filter"))
        except Exception:
            continue
        if "fault_active" in df.columns:
            df = df[df["fault_active"]]
        if "fault_type" in df.columns:
            types.update(df["fault_type"].dropna().unique().tolist())
    return sorted(types)


# ---------------------------------------------------------------------------
#  Collector: list of registry entries → one DataFrame
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Shared stratified-subsample primitive
# ---------------------------------------------------------------------------

def stratified_subsample(
    df: pd.DataFrame,
    max_rows: int,
    strat_col: str | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample ``df`` to at most ``max_rows``, stratified by ``strat_col``
    when possible.

    Shared by :func:`load_split` (used by exploration notebooks and,
    indirectly, anywhere that doesn't need to distinguish detection from
    classification) and :mod:`library.models.trainer`'s tabular training
    path (which needs to stratify by different columns depending on the
    task — ``"fault_type"`` for classification, ``"fault_active"`` for
    detection). Both previously carried their own copy of this same
    try/except-ValueError-fallback pattern; centralising it here means a
    future change to the fallback behaviour can't accidentally apply to
    only one of the two call sites.

    Parameters
    ----------
    df : pd.DataFrame
    max_rows : int
        Target row count. If ``len(df) <= max_rows``, ``df`` is returned
        unchanged (same object, no copy).
    strat_col : str or None
        Column to stratify by. If ``None`` or not present in ``df``, or if
        stratification fails (e.g. a class has a single member), falls
        back to a plain random sample.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Subsampled DataFrame with a fresh integer index (or the original
        ``df`` unchanged if it was already within ``max_rows``).
    """
    if len(df) <= max_rows:
        return df

    if strat_col and strat_col in df.columns:
        from sklearn.model_selection import train_test_split as _tts
        try:
            idx, _ = _tts(
                np.arange(len(df)), train_size=max_rows,
                stratify=df[strat_col].values, random_state=random_state,
            )
            log.info("  Stratified subsample (by %s) → %s rows",
                     strat_col, f"{max_rows:,}")
            return df.iloc[idx].reset_index(drop=True)
        except ValueError:
            log.info("  Stratification by %s failed (a class has < 2 "
                     "members) — falling back to random sample.", strat_col)

    log.info("  Random subsample → %s rows", f"{max_rows:,}")
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def load_split(
    entries: list[dict],
    cfg: BatchConfig,
    max_rows: int | None = None,
    lookback_window: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and concatenate data from a list of registry entries.

    The **single multi-entry loading function** for the library — used by
    the tabular training pipeline and by exploration notebooks.  Internally
    calls :func:`load_registry_entry` for every entry, guaranteeing
    identical preprocessing regardless of how the caller obtained the list.

    Replaces the former ``load_data_rf`` (directory scan + per-fault-type
    balanced sampling) and ``load_data_pt`` (directory scan + run_id column)
    with a single, registry-aware function that respects file-level split
    assignments and temporal year filters.

    Parameters
    ----------
    entries : list[dict]
        Registry entries to load.  Typically a split-filtered view, e.g.
        ``[e for e in registry if e["split"] == "train"]``, but can also be
        the full un-split registry for exploration notebooks.
    cfg : BatchConfig
    max_rows : int or None
        If set, subsample the concatenated DataFrame to this many rows via
        :func:`stratified_subsample`, stratified by ``fault_type`` when
        that column is present (so rare classes, e.g. ``curtailment`` with
        ~2k rows, are not eliminated). ``None`` loads everything.
    lookback_window : int
        Forwarded unchanged to :func:`load_registry_entry`.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Concatenated, daytime-filtered, feature-engineered data.
        Empty DataFrame (0 rows) is returned when all entries produce empty
        frames after filtering — callers should check ``len(data) > 0``.
    """
    frames = []
    for e in entries:
        df = load_registry_entry(e, cfg, lookback_window=lookback_window)
        if len(df):
            frames.append(df)

    if not frames:
        log.info("  load_split: no data from %d entries", len(entries))
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    log.info("  load_split: %s rows from %d entries",
             f"{len(data):,}", len(entries))

    if max_rows and len(data) > max_rows:
        strat_col = "fault_type" if "fault_type" in data.columns else None
        data = stratified_subsample(data, max_rows, strat_col, random_state)

    return data
