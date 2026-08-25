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
#  Memory estimation — makes "load everything" safe instead of a guess
# ---------------------------------------------------------------------------

def _available_memory_bytes() -> int:
    """Best-effort read of currently available system RAM, in bytes.

    Tries ``psutil`` first (most accurate — accounts for OS page cache
    that can be reclaimed). Falls back to parsing ``/proc/meminfo``'s
    ``MemAvailable`` on Linux if ``psutil`` isn't installed. If neither
    works (e.g. non-Linux without psutil), falls back to a conservative
    fixed assumption and logs a warning, so callers always get *some*
    number rather than crashing on the estimation step itself.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass

    fallback_gb = 4
    log.warning(
        "  Could not determine available system memory (psutil not "
        "installed and /proc/meminfo unavailable) — assuming a "
        "conservative %d GB. Install psutil for an accurate estimate: "
        "pip install psutil", fallback_gb,
    )
    return fallback_gb * (1024 ** 3)


def _registry_row_count(entries: list[dict]) -> int:
    """Total row count across every entry's parquet file, read from
    file metadata only — no columns are actually read, so this is cheap
    even across thousands of files. Used for the preflight size estimate
    below; ``sim_year_filter`` isn't applied here (that would require
    reading the sim_year column), so this is an upper bound, not exact,
    when entries carry a year filter.

    Returns 0 (rather than raising) if pyarrow is unavailable or every
    file fails to read — this function backs an optional safety warning,
    not a required code path, so it must never be able to crash a load
    that would otherwise have succeeded.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return 0

    total = 0
    for e in entries:
        try:
            total += pq.ParquetFile(e["path"]).metadata.num_rows
        except Exception:
            continue
    return total


def estimate_row_budget(
    entries: list[dict],
    cfg: BatchConfig,
    lookback_window: int = 12,
    mem_fraction: float = 0.4,
    safety_multiplier: float = 3.0,
    min_rows: int = 50_000,
    sample_size: int = 3,
    random_state: int = 42,
) -> int:
    """Compute a row cap that safely fits in currently available memory.

    This is what ``max_rows="auto"`` resolves to in :func:`load_split`.
    Rather than a hand-picked constant (which is either too conservative
    on a big machine or still crashes on a small one), this samples a
    few real files to measure actual per-row memory footprint of the
    fully feature-engineered DataFrame, reads how much RAM is currently
    available, and backs out how many rows fit.

    Parameters
    ----------
    entries : list[dict]
    cfg : BatchConfig
    lookback_window : int
        Forwarded to the sample loads so the measured per-row footprint
        reflects the same engineered columns the real load will produce.
    mem_fraction : float
        Fraction of *currently available* memory to budget for this
        load (default 0.4 — leaves headroom for everything else
        running: the notebook kernel itself, other loaded arrays, the
        eventual sklearn ``.fit()`` copy, matplotlib figures, etc.).
    safety_multiplier : float
        Extra derating applied on top of the raw per-row byte estimate
        (default 3x) to account for transient copies that exist only
        briefly but do count against peak RAM: the pandas frame itself,
        the ``build_feature_matrix`` numpy conversion, the stratified
        subsample's ``.iloc`` copy, and (for tabular training
        specifically) the deployment-model variant's sliced copy of the
        same array. Without this, a cap computed from raw DataFrame
        bytes alone would only be safe if nothing else ever touched the
        data — which every real call site here does.
    min_rows : int
        Floor on the computed budget, so a temporarily memory-starved
        machine doesn't get squeezed down to a handful of rows and
        produce a useless model. If the true safe budget is below this,
        the estimate is still returned as ``min_rows`` but a warning is
        logged — proceeding past this floor is the caller's call, not
        silently overridden here.
    sample_size : int
        Number of files to sample for the per-row byte measurement.
        Spread across the entry list (not just the first N) so the
        estimate isn't skewed by files that happen to sort first.
    random_state : int

    Returns
    -------
    int
        Safe row cap to pass as ``max_rows`` to :func:`load_split`.
    """
    if not entries:
        return min_rows

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(entries), size=min(sample_size, len(entries)), replace=False)
    sample_entries = [entries[i] for i in idx]

    total_bytes = 0
    total_rows = 0
    for e in sample_entries:
        try:
            df = load_registry_entry(e, cfg, lookback_window=lookback_window)
        except Exception:
            continue
        if len(df):
            total_bytes += int(df.memory_usage(deep=True).sum())
            total_rows += len(df)

    if total_rows == 0:
        log.warning("  estimate_row_budget: couldn't sample any rows — "
                   "falling back to min_rows=%s", f"{min_rows:,}")
        return min_rows

    bytes_per_row = (total_bytes / total_rows) * safety_multiplier
    available = _available_memory_bytes()
    budget = available * mem_fraction

    safe_rows = int(budget / bytes_per_row)
    safe_rows = max(safe_rows, min_rows)

    log.info(
        "  estimate_row_budget: %.0f bytes/row (incl. %.1fx safety margin), "
        "%s available, %.0f%% budgeted → cap = %s rows",
        bytes_per_row, safety_multiplier, f"{available / (1024**3):.1f} GB",
        mem_fraction * 100, f"{safe_rows:,}",
    )
    if safe_rows == min_rows and (budget / bytes_per_row) < min_rows:
        log.warning(
            "  estimate_row_budget: available memory only safely fits ~%s "
            "rows, below min_rows=%s — returning the floor anyway. "
            "Consider freeing memory or lowering min_rows if this cap is "
            "too small to be useful.",
            f"{int(budget / bytes_per_row):,}", f"{min_rows:,}",
        )
    return safe_rows


# ---------------------------------------------------------------------------
#  Dtype downcast — applied once, right at read time
# ---------------------------------------------------------------------------

# Columns worth converting to `category` when present as `object` dtype.
# Both are low-cardinality (a handful of fault types / run ids repeated
# across many thousands of rows), so pandas stores each unique string once
# and every row as a small integer code instead of a full Python string
# object — the usual categorical win, and a meaningful one here since
# `fault_type` and `run_id` are carried through every stage of the tabular
# accumulation buffer in :func:`load_split`.
_CATEGORICAL_COLS = ("fault_type", "run_id")


def _downcast_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 and low-cardinality string
    columns to category, immediately after a parquet read.

    Every physical quantity this library reads from SOLEY output
    (irradiance in W/m², power in kW, temperature in °C, angles in
    degrees, …) has far fewer significant digits of real-world meaning
    than float64 provides — the columns default to float64 only because
    that's pandas' default for floating-point parquet data, not because
    anything downstream needs that precision.
    :func:`~library.features.add_features` already casts every column it
    *creates* to float32; this closes the gap on the columns it *reads*.
    Previously a parquet file's original columns stayed at float64 for as
    long as they sat in a DataFrame, only getting downcast at the very
    end when :func:`~library.features.build_feature_matrix` built the
    final numpy array — so the accumulation buffer inside
    :func:`load_split` (the thing ``max_train_rows``/``max_test_rows``
    are supposed to bound) was carrying roughly double the memory it
    needed to, for its entire lifetime.

    Applying this once, here, means every caller of
    :func:`load_registry_entry` (tabular and sequence alike) inherits the
    smaller footprint for free, instead of each doing its own late,
    partial downcast.

    Parameters
    ----------
    df : pd.DataFrame
        A freshly read (and, if applicable, year-filtered) DataFrame.

    Returns
    -------
    pd.DataFrame
        Same data, smaller dtypes where safe to shrink.
    """
    float_cols = df.select_dtypes(include="float64").columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)

    for col in _CATEGORICAL_COLS:
        # Checked against "not already category" rather than "is object" —
        # pandas 3.x gives plain string columns a native `str` dtype by
        # default (not `object`), so an `== object` check would silently
        # never fire under that pandas version even though the column is
        # exactly the low-cardinality string data this is meant to catch.
        # astype("category") works the same regardless of which string
        # backend (object / python "string" / pyarrow "string" / pandas
        # 3.x "str") the column arrived as.
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    return df


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

    Also applies :func:`_downcast_numeric_dtypes` before returning, so
    every caller — including :func:`discover_fault_types`, which only
    reads two label columns — gets the smaller-footprint dtypes with no
    extra call site needed.

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

    return _downcast_numeric_dtypes(df)


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


# ---------------------------------------------------------------------------
#  Majority-class thinning — tabular-only redundancy reduction
# ---------------------------------------------------------------------------
#
# Why "majority", not "healthy"
# ------------------------------
# An earlier version of this hardcoded fault_active=False as the class to
# thin, on the (common, but not universal) assumption that faults are the
# rare case. That assumption is dataset-specific and can be flat wrong — in
# at least one real batch used with this library, fault_active=True was
# measured at ~80% of rows, i.e. the *opposite* imbalance. Redundancy comes
# from whichever condition happens to persist over a long contiguous stretch
# within a run — a multi-hour fault window is exactly as internally
# repetitive as a multi-hour healthy stretch. So this thins whichever class
# is locally larger in a given file, decided by counting each time, rather
# than assuming a fixed direction that only holds for some batches.
#
# The minority class (whichever it turns out to be) is always kept in full —
# it's still the rarer condition this file demonstrates, and a detection
# model needs both classes represented regardless of which one dominates.
#
# Why this lives in load_split, not load_registry_entry
# --------------------------------------------------------
# Thinning is applied here, inside the tabular collector, and deliberately
# NOT inside load_registry_entry (the atom shared with the sequence family).
# WindowDataset needs contiguous, evenly-spaced per-file timesteps to form
# meaningful temporal windows — thinning would silently gap the sequence
# models see, corrupting exactly the temporal structure LSTM/Hybrid rely on.
# Tabular rows are consumed independently of each other, so thinning is safe
# there and nowhere else in the library.

def _thin_majority_rows(
    df: pd.DataFrame,
    thin_factor: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly subsample whichever fault_active value is locally the
    majority within one file's data, keeping every minority-class row
    untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_registry_entry` for one file — must contain
        ``fault_active`` to do anything; returned unchanged otherwise.
    thin_factor : int
        Keep roughly ``1 / thin_factor`` of the majority-class rows. ``1``
        (or less) is a no-op, returned unchanged — this is the default
        everywhere thinning isn't explicitly requested, so exploration
        notebooks and evaluation always see the true, untouched
        distribution.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Every minority-class row, plus a random ``1/thin_factor`` sample
        of the majority-class rows. Which value of ``fault_active`` counts
        as "majority" is decided per file by comparing counts — not
        assumed. Row order is not preserved (irrelevant for the tabular
        family, which treats rows independently).
    """
    if thin_factor <= 1 or "fault_active" not in df.columns:
        return df

    true_rows = df[df["fault_active"]]
    false_rows = df[~df["fault_active"]]

    if len(true_rows) >= len(false_rows):
        majority, minority = true_rows, false_rows
    else:
        majority, minority = false_rows, true_rows

    if len(majority) == 0:
        return df

    keep_n = max(1, len(majority) // thin_factor)
    if keep_n >= len(majority):
        return df

    majority_thinned = majority.sample(n=keep_n, random_state=random_state)
    return pd.concat([minority, majority_thinned], ignore_index=True)


def load_split(
    entries: list[dict],
    cfg: BatchConfig,
    max_rows: int | str | None = None,
    lookback_window: int = 12,
    random_state: int = 42,
    mem_fraction: float = 0.4,
    majority_thin_factor: int = 1,
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
    max_rows : int, "auto", or None
        * ``int`` — subsample the concatenated DataFrame to this many
          rows via :func:`stratified_subsample`, stratified by
          ``fault_type`` when that column is present (so rare classes,
          e.g. ``curtailment`` with ~2k rows, are not eliminated).
        * ``"auto"`` — compute a safe cap from currently available
          system memory via :func:`estimate_row_budget` (sampling a few
          real files to measure actual per-row footprint), then use
          that as the cap. This is the recommended way to "use as much
          data as will safely fit" without picking a number by hand and
          without risking an out-of-memory crash — the cap adapts to
          the machine it's actually running on.
        * ``None`` — load everything, no cap. Before doing so, this
          logs a preflight estimate (cheap — reads only parquet
          metadata, no data) of the total rows involved and warns if
          the projected memory use looks likely to exceed what's
          currently available, so an eventual crash isn't a total
          surprise. This is a **warning, not a block** — the caller's
          explicit request to load everything is still honoured. If you
          want a hard guarantee against OOM instead, use ``"auto"``.
    lookback_window : int
        Forwarded unchanged to :func:`load_registry_entry`.
    random_state : int
    mem_fraction : float
        Forwarded to :func:`estimate_row_budget` when ``max_rows="auto"``,
        and used for the ``None``-case preflight warning threshold.
    majority_thin_factor : int
        Keep roughly ``1 / majority_thin_factor`` of each file's
        majority-``fault_active`` rows (whichever value — ``True`` or
        ``False`` — is more common *in that file*, decided per file, not
        assumed); the minority value is always kept in full. Default
        ``1`` — no thinning, every row is loaded exactly as before. This
        is a size/balance lever, not a class filter: unlike dropping
        "none" files outright (which would remove negative-class
        diversity a detection model needs), thinning only removes
        near-duplicate majority-class timesteps, applied per file before
        concatenation so the reduced volume is what flows into the
        shrink-buffer and final subsample below. Only meaningful for the
        ``fault_active`` detection task — for classification, healthy
        rows are dropped entirely downstream anyway (see
        :mod:`library.models.trainer`), so thinning has no effect there
        either way. Leave at ``1`` for exploration/evaluation call sites
        that need the true, untouched distribution (e.g.
        :mod:`01_data_exploration`'s fault prevalence chart) — only set
        above ``1`` at a training call site that has already decided it
        wants a lighter majority class. Don't push this so high that the
        resulting ratio overshoots past roughly 1:1 in the other
        direction — the point is trimming redundancy, not manufacturing a
        new imbalance in the opposite direction.

    Returns
    -------
    pd.DataFrame
        Concatenated, daytime-filtered, feature-engineered data.
        Empty DataFrame (0 rows) is returned when all entries produce empty
        frames after filtering — callers should check ``len(data) > 0``.

    Memory behaviour
    ----------------
    When ``max_rows`` is a number (whether given directly or resolved
    from ``"auto"``), peak memory is bounded to roughly
    ``shrink_factor * max_rows`` rows, **not** the size of the full
    concatenated dataset. Previously every entry's DataFrame was kept in
    a Python list until *all* entries had been read, and the ``max_rows``
    subsample was only applied to that fully-materialized whole — so a
    caller loading, say, a 200k-row sample out of a multi-hundred-file,
    multi-GB registry would still pay the full multi-GB peak RAM cost
    before ever shrinking down. Here, the buffer is periodically
    shrunk back down to ``max_rows`` (via the same stratified
    :func:`stratified_subsample` used for the final cap) once it grows
    past ``shrink_factor * max_rows``, so no more than a small multiple
    of the target sample size is ever held in memory at once, regardless
    of how many entries are in the registry. Passing ``max_rows=None``
    preserves the original full-materialize behaviour, since the caller
    has explicitly asked for everything — see the warning above.
    """
    if max_rows == "auto":
        max_rows = estimate_row_budget(
            entries, cfg, lookback_window=lookback_window,
            mem_fraction=mem_fraction, random_state=random_state,
        )
    elif max_rows is None:
        # Explicit "load everything" — this is the one case where a
        # dataset genuinely bigger than RAM *will* still crash (sklearn's
        # .fit() needs the full array resident at once; no amount of
        # loading cleverness changes that). Give visibility into that
        # risk before it happens rather than let an OS-level OOM kill
        # take the kernel down with no explanation.
        total_rows = _registry_row_count(entries)
        if total_rows:
            budget_rows = estimate_row_budget(
                entries, cfg, lookback_window=lookback_window,
                mem_fraction=mem_fraction, random_state=random_state,
                min_rows=1,  # just want the raw ratio here, not a floor
            )
            if total_rows > budget_rows:
                log.warning(
                    "  load_split: max_rows=None (load everything) requested "
                    "for %s rows across %d files, but only ~%s rows "
                    "safely fit in %.0f%% of currently available memory. "
                    "This may crash with an out-of-memory error. Consider "
                    "max_rows=\"auto\" to cap automatically, or an explicit "
                    "int, if that happens.",
                    f"{total_rows:,}", len(entries), f"{budget_rows:,}",
                    mem_fraction * 100,
                )

    # Only shrink incrementally when a cap was actually requested. The
    # factor (3x) leaves headroom so we're not constantly re-subsampling
    # after every single file, while still keeping peak memory a small,
    # constant multiple of max_rows instead of O(total dataset size).
    shrink_threshold = max_rows * 3 if max_rows else None

    # Process entries in a shuffled order, not registry order, whenever
    # incremental shrinking can kick in. Registry order comes from
    # sorted(data_dir.glob(...)) (see prepare_file_registry), and
    # _fault_type_from_filename derives each file's class from a
    # substring of its name — so files of the same fault_type routinely
    # sort into long contiguous runs (e.g. every "..._fault_shading_..."
    # file together, before any "..._fault_pid_..." file). Each
    # intermediate shrink only "sees" whatever has accumulated in the
    # buffer so far, so shrinking against a run of same-class files
    # produces a stratified sample of THAT RUN, not of the registry as a
    # whole — silently skewing the final class balance far from the true
    # dataset proportions (observed in practice: a 20% true fault rate
    # coming out at 50%+ faulty after shrinking). Shuffling first means
    # every intermediate buffer is itself an unbiased random subset of
    # the whole entry list, so a shrink at any point is statistically
    # equivalent to shrinking the full dataset. This has no effect when
    # max_rows is None (shrink_threshold is None, loop order doesn't
    # matter since nothing is subsampled until everything is loaded).
    order = list(range(len(entries)))
    if shrink_threshold:
        rng = np.random.RandomState(random_state)
        rng.shuffle(order)

    frames = []
    buffered_rows = 0

    for i, idx in enumerate(order):
        e = entries[idx]
        df = load_registry_entry(e, cfg, lookback_window=lookback_window)
        if majority_thin_factor > 1:
            df = _thin_majority_rows(df, majority_thin_factor, random_state)
        if len(df):
            frames.append(df)
            buffered_rows += len(df)

        if shrink_threshold and buffered_rows > shrink_threshold:
            merged = pd.concat(frames, ignore_index=True)
            strat_col = "fault_type" if "fault_type" in merged.columns else None
            merged = stratified_subsample(merged, max_rows, strat_col, random_state)
            frames = [merged]
            buffered_rows = len(merged)
            log.info("  load_split: shrank buffer to %s rows after %d/%d entries",
                     f"{buffered_rows:,}", i + 1, len(entries))

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
