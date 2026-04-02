"""
SOLEY ML Utilities -- Auto-configuration for any batch output
=============================================================

Shared utilities for ml_fault_demo.py and ml_sequence_demo.py.
All configuration is auto-detected from batch_config.json and
the parquet file schema. Nothing is hardcoded.

This module handles:
  - Reading batch configuration (array capacity, locations, etc.)
  - Auto-detecting and categorizing feature columns from parquet schema
  - Data loading with memory management
  - Feature engineering (performance ratio, time encoding)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("soley_ml")

# ===================================================================
#  COLUMN CATEGORIZATION
# ===================================================================

# Columns that are ALWAYS metadata (not features), regardless of name.
# These are structural columns used for indexing, splitting, and labeling.
_META_COLS = {
    "timestamp", "sim_day", "sim_year",
    "latitude", "longitude", "tilt_deg", "azimuth_deg", "dc_ac_ratio",
    "run_id",
}

# Columns that are fault labels (not features).
# Any column starting with "fault_" is a label column.
_LABEL_PREFIX = "fault_"

# Columns that are combo sub-fault tracking (not features).
_COMBO_PREFIX = "combo_"

# Columns that are noise metadata (not features).
_NOISE_COLS = {"noise_applied", "is_dropout", "timestamp_true"}

# Columns that real SCADA/monitoring systems can measure.
# Anything NOT in this set and NOT excluded is a "device physics" feature
# (only available from simulation, not from real sensors).
_SCADA_MEASURABLE = {
    # Irradiance (pyranometer / pyrheliometer)
    "poa_global_wm2", "ghi_wm2", "dni_wm2", "dhi_wm2",
    # Temperature (RTD / thermocouple)
    "ambient_temp_c", "cell_temp_c",
    # Wind (anemometer)
    "wind_speed_ms",
    # Precipitation (rain gauge or weather station)
    "precipitation_mm",
    # Power (CT / meter)
    "dc_power_kw", "ac_power_kw",
    # Solar geometry (computed from timestamp + location, universally available)
    "solar_elevation_deg", "aoi_deg",
    # Engineered features (computed from the above)
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "performance_ratio", "pr_deviation", "dc_ac_power_ratio", "power_step",
}

# Stress indicator prefix
_STRESS_PREFIX = "stress_"


class BatchConfig:
    """Auto-detected configuration from a SOLEY batch output directory.

    Reads batch_config.json and the first parquet file to discover:
    - Array capacity (kWp) for performance ratio
    - Location names and coordinates
    - Available feature columns, categorized into groups
    - Fault types present in the data

    Usage:
        cfg = BatchConfig("batch_output")
        print(cfg.array_kwp)           # 17.43
        print(cfg.locations)           # {(35.78, -78.64): "Raleigh, ..."}
        print(cfg.scada_features)      # ["poa_global_wm2", ...]
        print(cfg.device_features)     # ["voc_v", "ff", ...]
        print(cfg.stress_features)     # ["stress_cloud_edge", ...]
        print(cfg.all_feature_cols)    # union of all three
        print(cfg.load_cols)           # features + metadata + labels
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._read_config()
        self._discover_columns()

    def _read_config(self):
        """Read batch_config.json for array capacity and locations."""
        config_path = self.data_dir / "batch_config.json"

        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

            # Array capacity
            inst = cfg.get("installation_fixed", {})
            self.array_kwp = inst.get("array_power_kWp", None)

            # If not in config, try to compute from STC
            if self.array_kwp is None:
                stc = cfg.get("stc_parameters", {})
                area = inst.get("system_area_m2", 0)
                eff = stc.get("efficiency", 0)
                if area > 0 and eff > 0:
                    self.array_kwp = area * eff  # kWp = m2 * efficiency * 1kW/m2
                else:
                    self.array_kwp = None

            # Locations
            sweep = cfg.get("sweep_summary", {})
            self.locations = {}
            for loc in sweep.get("locations", []):
                lat = round(loc.get("lat", 0), 2)
                lon = round(loc.get("lon", 0), 2)
                # Extract short name (first part before first comma)
                full_name = loc.get("name", f"{lat}_{lon}")
                short_name = full_name.split(",")[0].strip()
                self.locations[(lat, lon)] = short_name

            self.config = cfg
            log.info("  Config: array=%.2f kWp, %d locations",
                     self.array_kwp or 0, len(self.locations))
        else:
            log.info("  No batch_config.json found -- using auto-detection")
            self.array_kwp = None
            self.locations = {}
            self.config = {}

    def _discover_columns(self):
        """Read a baseline parquet file schema to discover available columns.

        Categorizes columns into:
        - scada_features: measurable by real monitoring systems
        - device_features: only available from SOLEY simulation
        - stress_features: SOLEY physics-derived stress indicators
        - meta_cols: metadata for indexing/splitting
        - label_cols: fault labels for training targets
        - constant_cols: columns with zero variance (excluded from features)
        """
        import pyarrow.parquet as pq

        # Find a baseline file for schema discovery (prefer noisy — matches training)
        files = sorted(self.data_dir.glob("*_noisy.parquet"))
        if not files:
            files = sorted(self.data_dir.glob("*_clean.parquet"))
        if not files:
            files = sorted(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files in {self.data_dir}")

        # Prefer a baseline file (no "fault" in name) for clean schema
        baseline_files = [f for f in files if "fault" not in f.name]
        schema_file = baseline_files[0] if baseline_files else files[0]

        # Read schema (metadata only, no data)
        schema = pq.read_schema(schema_file)
        all_cols = set(schema.names)

        # Read a small sample to detect constant columns
        df_sample = pd.read_parquet(schema_file,
                                    columns=[n for n in schema.names])
        # Use first and last 1000 rows for variance check
        sample = pd.concat([df_sample.head(1000), df_sample.tail(1000)])

        constant_cols = set()
        for col in sample.columns:
            try:
                if sample[col].nunique(dropna=False) <= 1:
                    constant_cols.add(col)
            except TypeError:
                pass

        # Categorize columns
        self.meta_cols = _META_COLS & all_cols
        self.label_cols = {c for c in all_cols if c.startswith(_LABEL_PREFIX)}
        self.combo_cols = {c for c in all_cols if c.startswith(_COMBO_PREFIX)}
        self.noise_cols = _NOISE_COLS & all_cols
        self.constant_cols = constant_cols

        # Columns excluded from features
        excluded = (self.meta_cols | self.label_cols | self.combo_cols
                    | self.noise_cols | {"timestamp", "run_id"}
                    | {c for c in all_cols if c.endswith("_true")})

        # Feature columns = everything not excluded and not constant
        feature_candidates = all_cols - excluded

        # Categorize features
        self.stress_features = sorted(
            c for c in feature_candidates
            if c.startswith(_STRESS_PREFIX) and c not in constant_cols
        )
        self.scada_features = sorted(
            c for c in feature_candidates
            if c in _SCADA_MEASURABLE and c not in constant_cols
        )
        self.device_features = sorted(
            c for c in feature_candidates
            if c not in _SCADA_MEASURABLE
            and not c.startswith(_STRESS_PREFIX)
            and c not in constant_cols
        )

        # All features combined
        self.all_feature_cols = (
            self.scada_features + self.device_features + self.stress_features
        )

        # Columns to load from parquet (features + metadata + labels needed)
        self.load_cols = sorted(set(
            self.all_feature_cols
            + list(self.meta_cols & all_cols)
            + list(self.label_cols & all_cols)
            + ["timestamp"]
        ) & all_cols)

        # If array_kwp wasn't in config, estimate from data
        if self.array_kwp is None and "dc_power_kw" in all_cols:
            max_dc = df_sample["dc_power_kw"].max()
            self.array_kwp = float(max_dc) * 1.05  # rough estimate
            log.info("  Estimated array capacity: %.2f kWp (from max DC power)",
                     self.array_kwp)

        log.info("  Columns: %d SCADA, %d device physics, %d stress, "
                 "%d constant (excluded), %d total features",
                 len(self.scada_features), len(self.device_features),
                 len(self.stress_features), len(constant_cols),
                 len(self.all_feature_cols))

    def get_location_name(self, lat, lon):
        """Get human-readable location name from coordinates."""
        key = (round(lat, 2), round(lon, 2))
        if key in self.locations:
            return self.locations[key]
        # Try with less precision
        for (k_lat, k_lon), name in self.locations.items():
            if abs(k_lat - lat) < 0.05 and abs(k_lon - lon) < 0.05:
                return name
        return f"{lat:.2f}_{lon:.2f}"

    def get_loc_id(self, lat, lon):
        """Get a stable location ID string from coordinates."""
        return f"{round(lat, 2)}_{round(lon, 2)}"


# ===================================================================
#  FEATURE ENGINEERING
# ===================================================================

def add_features(df, array_kwp, poa_threshold=10):
    """Standard feature engineering for ML training.

    - Filters to daytime only (POA > threshold)
    - Adds cyclical time encoding (hour, day-of-year)
    - Adds performance ratio (IEC 61724)

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with at least poa_global_wm2, ac_power_kw, timestamp.
    array_kwp : float
        Array DC capacity in kWp (from BatchConfig).
    poa_threshold : float
        Minimum POA to include (default: 10 W/m2).

    Returns
    -------
    pd.DataFrame with added features, daytime only.
    """
    df = df[df["poa_global_wm2"] > poa_threshold].copy()

    # Cyclical time features
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0).astype(np.float32)

    # Performance ratio (IEC 61724)
    if array_kwp and array_kwp > 0:
        poa_normalized = df["poa_global_wm2"] / 1000.0
        poa_normalized = poa_normalized.replace(0, np.nan)
        df["performance_ratio"] = (
            (df["ac_power_kw"] / array_kwp) / poa_normalized
        ).astype(np.float32)
        df["performance_ratio"] = df["performance_ratio"].fillna(0).clip(-1, 2)
    else:
        # Fallback: raw power/irradiance ratio (not true PR, but usable)
        poa_kw = (df["poa_global_wm2"] / 1000.0).replace(0, np.nan)
        df["performance_ratio"] = (
            df["ac_power_kw"] / poa_kw
        ).astype(np.float32)
        df["performance_ratio"] = df["performance_ratio"].fillna(0).clip(-100, 100)

    # Rolling PR deviation — distinguishes persistent events (soiling,
    # progressive faults) from transient events (clouds, brief outages).
    # PR_deviation = current PR / rolling-mean PR over last 7 days.
    # Soiling: sudden step-down that persists (deviation << 1 for days).
    # Cloud: PR drops but rolling mean stays stable (deviation recovers quickly).
    # Progressive fault: slow decline (deviation drifts gradually below 1).
    pr = df["performance_ratio"]
    # Window: 7 days of daytime timesteps (~84 for hourly, ~588 for 5-min)
    # Use 500 as a robust approximation for 5-min data
    window = min(500, max(12, len(df) // 100))
    rolling_mean = pr.rolling(window=window, min_periods=max(1, window // 4),
                              center=False).mean()
    deviation = pr / rolling_mean.replace(0, np.nan)
    df["pr_deviation"] = deviation.fillna(1.0).clip(0, 3).astype(np.float32)

    # DC/AC power ratio — distinguishes curtailment (normal DC, capped AC)
    # from MPPT failure (reduced DC, proportional AC).
    # Curtailment: dc_ac_power_ratio >> 1 (DC is fine, AC is capped).
    # MPPT failure: dc_ac_power_ratio ~ normal (both reduced proportionally).
    ac = df["ac_power_kw"].replace(0, np.nan)
    df["dc_ac_power_ratio"] = (df["dc_power_kw"] / ac).fillna(1.0).clip(0, 5).astype(np.float32)

    # Power step — max absolute PR change within a short window.
    # Cell crack: large single-step drop (power_step >> 0).
    # Solder fatigue: no sudden step (power_step ~ 0).
    # Computed as max |delta PR| over last 12 timesteps (1 hour at 5-min).
    pr_diff = pr.diff().abs()
    df["power_step"] = pr_diff.rolling(window=12, min_periods=1).max().fillna(0).clip(0, 2).astype(np.float32)

    log.info("After daytime filter: %s rows", f"{len(df):,}")
    return df


# ===================================================================
#  DATA LOADING (RF -- per-file subsampling)
# ===================================================================

def load_data_rf(data_dir, cfg, max_rows=None, location_filter=None):
    """Load batch output for RF training with fault-type-balanced subsampling.

    Groups files by fault type, allocates equal row budget per type,
    then subsamples within each group. This ensures the model sees
    balanced representation of all fault types regardless of how many
    sweep variants each fault has.

    Parameters
    ----------
    data_dir : str or Path
    cfg : BatchConfig
    max_rows : int or None
        If set, balance across fault types to stay near this limit.
    location_filter : str or None
        Only load files matching this lat_lon string.

    Returns
    -------
    pd.DataFrame
    """
    import pyarrow.parquet as pq
    from collections import defaultdict

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_noisy.parquet"))
    if not files:
        files = sorted(data_dir.glob("*_clean.parquet"))
    if not files:
        log.error("No parquet files in %s", data_dir)
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    if location_filter:
        files = [f for f in files if location_filter in f.name]

    # Group files by fault type for balanced loading
    fault_groups = defaultdict(list)
    for f in files:
        name = f.stem
        if "_fault_" in name:
            # Extract fault type from filename: ..._fault_TYPE_af...
            after_fault = name.split("_fault_")[1]
            # Fault type ends at _af or _aging
            for sep in ("_af", "_aging"):
                if sep in after_fault:
                    ft = after_fault.split(sep)[0]
                    break
            else:
                ft = after_fault
            fault_groups[ft].append(f)
        elif "_combo_" in name or "combo" in name:
            fault_groups["combo"].append(f)
        else:
            fault_groups["none"].append(f)

    n_types = len(fault_groups)
    log.info("Loading %d files from %s (%d fault types) ...",
             len(files), data_dir, n_types)

    # Allocate row budget per fault type
    if max_rows is not None:
        rows_per_type = max_rows // n_types
        log.info("Balanced subsampling: ~%s rows per fault type (%d types), "
                 "%s total",
                 f"{rows_per_type:,}", n_types, f"{max_rows:,}")
    else:
        rows_per_type = None

    frames = []
    rng = np.random.RandomState(42)
    load_cols = cfg.load_cols
    files_loaded = 0

    for ft in sorted(fault_groups.keys()):
        ft_files = fault_groups[ft]

        # Determine per-file sample size for this fault type
        ft_rows_per_file = None
        if rows_per_type is not None:
            sample_nrows = pq.read_metadata(ft_files[0]).num_rows
            ft_total_est = sample_nrows * len(ft_files)
            if ft_total_est > rows_per_type:
                ft_rows_per_file = max(500, rows_per_type // len(ft_files))

        for f in ft_files:
            try:
                df = pd.read_parquet(f, columns=load_cols)
            except Exception:
                schema_cols = set(pq.read_schema(f).names)
                cols = [c for c in load_cols if c in schema_cols]
                df = pd.read_parquet(f, columns=cols)

            if ft_rows_per_file is not None and len(df) > ft_rows_per_file:
                frac = ft_rows_per_file / len(df)
                if "fault_active" in df.columns:
                    idx = df.groupby("fault_active", group_keys=False).apply(
                        lambda g: g.sample(frac=frac, random_state=rng),
                    ).index
                    df = df.loc[idx]
                else:
                    df = df.sample(n=ft_rows_per_file, random_state=rng)

            frames.append(df)
            files_loaded += 1

            if files_loaded % 25 == 0 or files_loaded == len(files):
                log.info("  ... loaded %d / %d files", files_loaded, len(files))

    data = pd.concat(frames, ignore_index=True)
    log.info("Total: %s rows, %d columns", f"{len(data):,}", len(data.columns))

    if "fault_type" in data.columns:
        log.info("")
        log.info("Fault distribution:")
        vc = data["fault_type"].value_counts()
        for ft, cnt in vc.items():
            log.info("  %-30s %10s  (%5.1f%%)", ft, f"{cnt:,}",
                     100.0 * cnt / len(data))

    return data


# ===================================================================
#  DATA LOADING (PyTorch -- run-limited, contiguous timesteps)
# ===================================================================

def load_data_pt(data_dir, cfg, max_runs=None, location_filter=None):
    """Load batch output for PyTorch training with run limiting.

    Unlike load_data_rf, this loads FULL files (contiguous timesteps
    needed for windowing) but limits the NUMBER of files.

    Parameters
    ----------
    data_dir : str or Path
    cfg : BatchConfig
    max_runs : int or None
        Max files to load. Balanced across fault types.
    location_filter : str or None
        Only load files matching this lat_lon string.

    Returns
    -------
    pd.DataFrame with a 'run_id' column.
    """
    import pyarrow.parquet as pq
    from collections import defaultdict

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_noisy.parquet"))
    if not files:
        files = sorted(data_dir.glob("*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    if location_filter:
        files = [f for f in files if location_filter in f.name]

    # Limit runs: balanced across fault types
    if max_runs is not None and len(files) > max_runs:
        groups = defaultdict(list)
        for f in files:
            name = f.stem
            if "_fault_" in name:
                after_fault = name.split("_fault_")[1]
                parts = after_fault.split("_af")
                ft = parts[0] if len(parts) > 1 else after_fault.split("_aging")[0]
            else:
                ft = "none"
            groups[ft].append(f)

        n_types = len(groups)
        per_type = max(1, max_runs // n_types)
        selected = []
        for ft in sorted(groups.keys()):
            selected.extend(groups[ft][:per_type])
        files = sorted(selected)[:max_runs]
        log.info("Selected %d files (%d per fault type, %d types)",
                 len(files), per_type, n_types)

    log.info("Loading %d files ...", len(files))

    frames = []
    load_cols = cfg.load_cols

    for i, f in enumerate(files):
        parts = f.stem.split("_")
        run_id = parts[0]

        try:
            df = pd.read_parquet(f, columns=load_cols)
        except Exception:
            schema_cols = set(pq.read_schema(f).names)
            cols = [c for c in load_cols if c in schema_cols]
            df = pd.read_parquet(f, columns=cols)

        df["run_id"] = run_id
        frames.append(df)

        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            log.info("  ... %d / %d", i + 1, len(files))

    data = pd.concat(frames, ignore_index=True)
    log.info("Loaded %s rows, %d runs", f"{len(data):,}",
             data["run_id"].nunique())
    return data


# ===================================================================
#  PLOTTING HELPERS
# ===================================================================

def plot_confusion(y_true, y_pred, class_names, title, path):
    """Plot a confusion matrix with counts annotated in each cell."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    unique_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 0.9)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=9 if n > 6 else 11)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{title} -- Confusion Matrix", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)


def plot_importance(model, feature_names, title, path,
                    scada_features, device_features, top_n=20):
    """Horizontal bar chart of feature importances, color-coded by group."""
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    top_n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))

    scada_set = set(scada_features)
    device_set = set(device_features)

    colors = []
    for i in indices:
        name = feature_names[i]
        if name.startswith("stress_"):
            colors.append("#f59e0b")   # amber = stress
        elif name in device_set:
            colors.append("#10b981")   # green = device physics
        else:
            colors.append("#2563eb")   # blue = SCADA

    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini impurity reduction)", fontsize=11)
    ax.set_title(f"{title} -- Top {top_n} Features\n"
                 "(blue=SCADA  green=device physics  amber=stress)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)
