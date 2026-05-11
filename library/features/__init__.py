"""
library.features
=================

Feature engineering for SOLEY PV time-series data.

  add_features(df, array_kwp) — main entry point:
      * Daytime filter (POA > threshold)
      * Cyclical time encoding  (hour_sin/cos, doy_sin/cos)
      * Performance ratio (IEC 61724)
      * Rolling PR deviation  (persistent vs. transient faults)
      * DC/AC power ratio     (curtailment vs. MPPT failure)
      * Power step            (sudden vs. gradual faults)

  build_feature_matrix(df, feature_names) — extract clean numpy array.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Main feature-engineering pipeline
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame, array_kwp: float | None,
                 poa_threshold: float = 10.0,
                 lookback_window: int = 12) -> pd.DataFrame:
    """Standard feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SOLEY parquet data. Must contain at least
        ``poa_global_wm2``, ``ac_power_kw``, ``dc_power_kw``,
        and ``timestamp``.
    array_kwp : float or None
        Array DC capacity in kWp.  Used to compute the performance ratio.
        If ``None``, a fallback power/irradiance ratio is used instead.
    poa_threshold : float
        Minimum plane-of-array irradiance (W/m²) to include a row.
        Rows with POA ≤ threshold are night-time and are dropped.
    lookback_window : int
        Number of recent daytime timesteps used for both the rolling
        PR-deviation baseline and the power-step maximum.  Must be the
        same value used as ``window_size`` in :class:`~library.models.dataset.WindowDataset`
        so that both the Random Forest and the LSTM/hybrid architectures
        see identical historical context.  Default: ``12`` (≈ 1 h at 5-min
        resolution).

    Returns
    -------
    pd.DataFrame
        Daytime-filtered copy of ``df`` with the new feature columns
        added in-place (as float32).

    New columns
    -----------
    hour_sin, hour_cos     — cyclical hour-of-day encoding
    doy_sin, doy_cos       — cyclical day-of-year encoding
    performance_ratio      — IEC 61724 PR  (AC output / irradiance-normalised capacity)
    pr_deviation           — current PR / rolling ``lookback_window``-step mean PR
                             (distinguishes soiling from transient clouds)
    dc_ac_power_ratio      — DC / AC power
                             (curtailment: ratio >> 1; MPPT failure: ratio ≈ 1)
    power_step             — max |ΔPR| over last ``lookback_window`` timesteps
                             (cell crack: large step; solder fatigue: no step)
    """
    # --- Daytime filter ---------------------------------------------------
    df = df[df["poa_global_wm2"] > poa_threshold].copy()

    # --- Cyclical time encoding -------------------------------------------
    ts   = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy  = ts.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["doy_sin"]  = np.sin(2 * np.pi * doy  / 365.0).astype(np.float32)
    df["doy_cos"]  = np.cos(2 * np.pi * doy  / 365.0).astype(np.float32)

    # --- Performance ratio (IEC 61724) ------------------------------------
    if array_kwp and array_kwp > 0:
        poa_norm = (df["poa_global_wm2"] / 1000.0).replace(0, np.nan)
        df["performance_ratio"] = (
            (df["ac_power_kw"] / array_kwp) / poa_norm
        ).fillna(0).clip(-1, 2).astype(np.float32)
    else:
        poa_kw = (df["poa_global_wm2"] / 1000.0).replace(0, np.nan)
        df["performance_ratio"] = (
            df["ac_power_kw"] / poa_kw
        ).fillna(0).clip(-100, 100).astype(np.float32)

    # --- Rolling PR deviation ---------------------------------------------
    # Window = lookback_window daytime steps (matches WindowDataset.window_size).
    pr           = df["performance_ratio"]
    rolling_mean = pr.rolling(window=lookback_window,
                              min_periods=max(1, lookback_window // 4),
                              center=False).mean()
    deviation = pr / rolling_mean.replace(0, np.nan)
    df["pr_deviation"] = deviation.fillna(1.0).clip(0, 3).astype(np.float32)

    # --- DC / AC power ratio ----------------------------------------------
    ac = df["ac_power_kw"].replace(0, np.nan)
    df["dc_ac_power_ratio"] = (
        df["dc_power_kw"] / ac
    ).fillna(1.0).clip(0, 5).astype(np.float32)

    # --- Power step (max |ΔPR| over lookback_window steps) ----------------
    pr_diff = pr.diff().abs()
    df["power_step"] = (
        pr_diff.rolling(window=lookback_window, min_periods=1).max()
        .fillna(0).clip(0, 2).astype(np.float32)
    )

    log.info("After daytime filter: %s rows", f"{len(df):,}")
    return df


# ---------------------------------------------------------------------------
#  Feature matrix helper
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract a clean float32 feature matrix from a DataFrame.

    Missing columns are filled with 0.  Inf / NaN values are
    replaced with 0 before returning.

    Parameters
    ----------
    df : pd.DataFrame
    feature_names : list[str]

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    used_features : list[str]
        The subset of ``feature_names`` that were actually present in df.
    """
    present = [c for c in feature_names if c in df.columns]
    missing = set(feature_names) - set(present)
    if missing:
        log.info("  (missing columns filled with 0: %s)",
                 ", ".join(sorted(missing)))

    X = (df[present]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0)
         .values
         .astype(np.float32))
    return X, present


__all__ = ["add_features", "build_feature_matrix"]
