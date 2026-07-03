"""
library.config.batch_config
============================

Auto-detects everything the ML pipeline needs from a SOLEY batch output
directory:

  * Array DC capacity (kWp) for performance-ratio computation.
  * Site coordinates and human-readable location names.
  * All available parquet columns, categorised into:
      - scada_features   : measurable by real SCADA / monitoring systems
      - device_features  : available only from SOLEY simulation physics
      - stress_features  : SOLEY-derived stress indicators (stress_*)
      - meta_cols        : indexing / splitting columns
      - label_cols       : fault labels (fault_*)
  * load_cols: the minimal set of columns to read from parquet files.

Nothing is hard-coded. BatchConfig works with any SOLEY batch output.
"""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("library")

# ---------------------------------------------------------------------------
#  Column classification rules
# ---------------------------------------------------------------------------

_META_COLS = {
    "timestamp", "sim_day", "sim_year",
    "latitude", "longitude", "tilt_deg", "azimuth_deg", "dc_ac_ratio",
    "run_id",
}

_LABEL_PREFIX   = "fault_"
_COMBO_PREFIX   = "combo_"
_STRESS_PREFIX  = "stress_"
_NOISE_COLS     = {"noise_applied", "is_dropout", "timestamp_true"}

# Engineered features derived solely from SCADA signals (added by
# add_features(), never present as raw parquet columns). Defined once here
# and referenced both by _SCADA_MEASURABLE (so a column of this name is
# classified as SCADA-derived if it were ever found in raw data) and by
# feature_set() (to build the "scada" feature list) — previously this list
# was duplicated verbatim in both places, risking drift if a new engineered
# feature were ever added to only one of them.
_ENGINEERED_FEATURES = (
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "performance_ratio", "pr_deviation", "dc_ac_power_ratio", "power_step",
)

# Columns that a real SCADA / monitoring system can measure directly.
# Everything else is a "device physics" column (simulation-only).
_SCADA_MEASURABLE = {
    "poa_global_wm2", "ghi_wm2", "dni_wm2", "dhi_wm2",
    "ambient_temp_c", "cell_temp_c",
    "wind_speed_ms",
    "precipitation_mm",
    "dc_power_kw", "ac_power_kw",
    "solar_elevation_deg", "aoi_deg",
} | set(_ENGINEERED_FEATURES)

# Columns that are simulation-only "god-mode" outputs — never available in a
# real installation and MUST NOT be used as model inputs.
_GOD_MODE_COLS = {
    "voc_v",
    "jsc_a_m2",
    "ff",
    "vmpp_v",
    "jmpp_a_m2",
    "detailed_balance_efficiency_pct",
}


class BatchConfig:
    """Auto-detected configuration for a SOLEY batch output directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to the SOLEY batch output folder containing parquet files
        and (optionally) a ``batch_config.json``.

    Attributes
    ----------
    array_kwp : float or None
        Array DC capacity in kWp. Used for performance-ratio computation.
    locations : dict
        ``{(lat, lon): "City, Country"}`` for every site in the batch.
    scada_features : list[str]
        Feature columns measurable by real monitoring hardware.
    device_features : list[str]
        Feature columns available only from SOLEY simulation.
    stress_features : list[str]
        Physics-derived stress indicator columns (``stress_*``).
    all_feature_cols : list[str]
        Union of the three feature lists above.
    meta_cols : set[str]
        Metadata columns (latitude, longitude, sim_year, …).
    label_cols : set[str]
        Fault label columns (fault_active, fault_type, …).
    load_cols : list[str]
        Minimal column list to pass to ``pd.read_parquet``.

    Examples
    --------
    >>> cfg = BatchConfig("batch_output")
    >>> print(cfg.array_kwp)        # 17.43
    >>> print(cfg.locations)        # {(35.78, -78.64): "Raleigh"}
    >>> print(cfg.scada_features)   # ["poa_global_wm2", "dc_power_kw", ...]
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._read_config()
        self._discover_columns()

    # ------------------------------------------------------------------
    #  Step 1: read batch_config.json
    # ------------------------------------------------------------------

    def _read_config(self):
        config_path = self.data_dir / "batch_config.json"

        if config_path.exists():
            with open(config_path) as fh:
                cfg = json.load(fh)

            inst = cfg.get("installation_fixed", {})
            self.array_kwp = inst.get("array_power_kWp", None)

            # Fallback: compute from STC parameters
            if self.array_kwp is None:
                stc  = cfg.get("stc_parameters", {})
                area = inst.get("system_area_m2", 0)
                eff  = stc.get("efficiency", 0)
                self.array_kwp = (area * eff) if (area > 0 and eff > 0) else None

            # Locations
            sweep = cfg.get("sweep_summary", {})
            self.locations: dict[tuple[float, float], str] = {}
            for loc in sweep.get("locations", []):
                lat = round(loc.get("lat", 0), 2)
                lon = round(loc.get("lon", 0), 2)
                full_name  = loc.get("name", f"{lat}_{lon}")
                short_name = full_name.split(",")[0].strip()
                self.locations[(lat, lon)] = short_name

            self.config = cfg
            log.info("  Config: array=%.2f kWp, %d location(s)",
                     self.array_kwp or 0, len(self.locations))
        else:
            log.info("  No batch_config.json — using auto-detection.")
            self.array_kwp = None
            self.locations = {}
            self.config    = {}

    # ------------------------------------------------------------------
    #  Step 2: discover and categorise columns from parquet schema
    # ------------------------------------------------------------------

    def _discover_columns(self):
        """Read the schema of one baseline parquet file and classify columns."""
        import pyarrow.parquet as pq

        files = sorted(self.data_dir.glob("*_noisy.parquet"))
        if not files:
            files = sorted(self.data_dir.glob("*_clean.parquet"))
        if not files:
            files = sorted(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        # Prefer a baseline (no-fault) file for schema discovery
        baseline = [f for f in files if "fault" not in f.name]
        schema_file = baseline[0] if baseline else files[0]

        schema   = pq.read_schema(schema_file)
        all_cols = set(schema.names)

        # Small sample to detect zero-variance (constant) columns
        df_sample = pd.read_parquet(schema_file)
        sample    = pd.concat([df_sample.head(1000), df_sample.tail(1000)])

        constant_cols: set[str] = set()
        for col in sample.columns:
            try:
                if sample[col].nunique(dropna=False) <= 1:
                    constant_cols.add(col)
            except TypeError:
                pass

        self.meta_cols    = _META_COLS & all_cols
        self.label_cols   = {c for c in all_cols if c.startswith(_LABEL_PREFIX)}
        self.combo_cols   = {c for c in all_cols if c.startswith(_COMBO_PREFIX)}
        self.noise_cols   = _NOISE_COLS & all_cols
        self.constant_cols = constant_cols

        excluded = (
            self.meta_cols | self.label_cols | self.combo_cols
            | self.noise_cols | {"timestamp", "run_id"}
            | {c for c in all_cols if c.endswith("_true")}
            | _GOD_MODE_COLS          # simulation-only; never available in real installs
        )
        feature_candidates = all_cols - excluded

        self.stress_features: list[str] = sorted(
            c for c in feature_candidates
            if c.startswith(_STRESS_PREFIX) and c not in constant_cols
        )
        self.scada_features: list[str] = sorted(
            c for c in feature_candidates
            if c in _SCADA_MEASURABLE and c not in constant_cols
        )
        self.device_features: list[str] = sorted(
            c for c in feature_candidates
            if c not in _SCADA_MEASURABLE
            and not c.startswith(_STRESS_PREFIX)
            and c not in constant_cols
        )
        self.all_feature_cols: list[str] = (
            self.scada_features + self.device_features + self.stress_features
        )

        # Minimal load list: features + metadata + labels
        self.load_cols: list[str] = sorted(
            set(self.all_feature_cols)
            | (self.meta_cols & all_cols)
            | (self.label_cols & all_cols)
            | {"timestamp"}
        ) 
        self.load_cols = [c for c in self.load_cols if c in all_cols]

        # Estimate array capacity from data if not in config
        if self.array_kwp is None and "dc_power_kw" in all_cols:
            self.array_kwp = float(df_sample["dc_power_kw"].max()) * 1.05
            log.info("  Estimated array capacity: %.2f kWp (from max DC power)",
                     self.array_kwp)

        god_mode_found = _GOD_MODE_COLS & all_cols
        if god_mode_found:
            log.warning(
                "  God-mode columns found in data and excluded from all feature sets: %s",
                sorted(god_mode_found),
            )
        log.info(
            "  Columns: %d SCADA, %d device physics, %d stress, "
            "%d constant (excluded), %d god-mode (excluded), %d total features",
            len(self.scada_features), len(self.device_features),
            len(self.stress_features), len(constant_cols),
            len(god_mode_found), len(self.all_feature_cols),
        )

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------

    def get_location_name(self, lat: float, lon: float) -> str:
        """Human-readable name for a (lat, lon) pair."""
        key = (round(lat, 2), round(lon, 2))
        if key in self.locations:
            return self.locations[key]
        for (k_lat, k_lon), name in self.locations.items():
            if abs(k_lat - lat) < 0.05 and abs(k_lon - lon) < 0.05:
                return name
        return f"{lat:.2f}_{lon:.2f}"

    def get_loc_id(self, lat: float, lon: float) -> str:
        """Stable string ID for a (lat, lon) pair."""
        return f"{round(lat, 2)}_{round(lon, 2)}"

    def feature_set(self, mode: str = "full") -> list[str]:
        """Return a named feature subset.

        Parameters
        ----------
        mode : {"full", "scada", "scada+stress", "scada+device"}
        """
        eng = list(_ENGINEERED_FEATURES)
        scada = list(self.scada_features) + eng

        if mode == "scada":
            return scada
        if mode == "scada+stress":
            return scada + list(self.stress_features)
        if mode == "scada+device":
            return scada + list(self.device_features)
        # full
        return scada + list(self.device_features) + list(self.stress_features)

    def __repr__(self) -> str:
        return (
            f"BatchConfig("
            f"array_kwp={self.array_kwp:.2f}, "
            f"locations={len(self.locations)}, "
            f"features={len(self.all_feature_cols)})"
        )
