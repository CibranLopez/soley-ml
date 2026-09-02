"""
library.config.batch_config
============================

Auto-detects everything the ML pipeline needs from a SOLEY batch output
directory:

  * Array DC capacity (kWp) for performance-ratio computation.
  * Site coordinates and human-readable location names.
  * All available parquet columns, categorised into:
      - scada_features    : measurable by real SCADA / monitoring systems
      - iv_curve_features  : real IV-curve-tracer outputs (Voc, Jsc, FF, ...) —
                            not baseline SCADA, but a real capability on
                            advanced/smart-grid-connected installations
      - stress_features   : weather-derived stress indicators (stress_*) —
                            deterministic, computed from weather data, but
                            not currently exposed by this pipeline's SCADA
      - meta_cols         : indexing / splitting columns
      - label_cols        : fault labels (fault_*)
  * load_cols: the minimal set of columns to read from parquet files.

Column classification is DEFAULT-DENY: a column only becomes a usable
feature if it's explicitly recognised as SCADA-measurable, an IV-curve
output, or stress-prefixed. Everything else — including any new,
not-yet-seen column in a future SOLEY batch — is excluded and logged as
"uncategorized" rather than silently absorbed into a catch-all feature
bucket. This replaced an earlier "device_features" catch-all (anything
not otherwise classified) that had silently been including simulator-
internal state columns — aging_factor, soiling_factor, vulnerability,
and similar — as model inputs. These directly cause the very power
deviations the model is meant to learn to detect from indirect,
real-world-observable telemetry; using them as features let a model
shortcut around learning that signal, and would have provided zero
predictive value on a real (non-simulated) installation. Confirmed via
the domain owner: categories 4-7 of the SOLEY schema (geometry/location
identifiers, loss & correction factors, fault labels, and simulation
truth/noise columns) are metadata and targets, never model inputs — see
_SIMULATION_INTERNAL_COLS below for the specific columns this excludes.

Nothing else is hard-coded. BatchConfig works with any SOLEY batch output.
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

# Columns a real SCADA / monitoring system can measure or directly compute
# from geometry + time (solar position and angle of incidence are standard
# outputs of any real PV monitoring/tracking platform, not simulation
# ground truth). Everything else needs its own explicit category below —
# there is deliberately no catch-all "anything else is SCADA-adjacent"
# rule; see the module docstring for why.
#
# NOTE: efficiency_pct is intentionally NOT included here even though it
# sounds like a SCADA quantity — confirmed with the domain owner that real
# monitoring systems don't report it directly (only dc/ac power, from which
# it COULD be derived), and in this simulator it's computed by the physics
# engine. It's classified under _IV_CURVE_COLS's spirit but doesn't fit
# that name either, so it gets its own tiny set below rather than being
# mis-filed into _SCADA_MEASURABLE.
_SCADA_MEASURABLE = {
    "poa_global_wm2", "ghi_wm2", "dni_wm2", "dhi_wm2",
    "ambient_temp_c", "cell_temp_c",
    "wind_speed_ms",
    "precipitation_mm",
    "dc_power_kw", "ac_power_kw",
    "solar_elevation_deg", "aoi_deg",
} | set(_ENGINEERED_FEATURES)

# Real IV-curve-tracer outputs. Not part of baseline SCADA telemetry — but
# unlike DETAILED_BALANCE_EFFICIENCY below (a theoretical/idealized
# quantity with no real-world sensor equivalent), periodic IV-curve tracing
# is a real capability on modern smart-grid-connected installations. Legit
# "full"/research-tier features; not "deployment"-tier. efficiency_pct is
# simulator-computed (see _SCADA_MEASURABLE note above) but conceptually
# belongs with this tier — derivable from an IV curve, not raw SCADA.
_IV_CURVE_COLS = {
    "voc_v", "jsc_a_m2", "ff", "vmpp_v", "jmpp_a_m2",
    "efficiency_pct",
}

# Purely theoretical physics-engine output — a thermodynamic upper bound,
# not something any real installation, however advanced, could measure.
# Always excluded from every feature_set() mode.
_THEORETICAL_ONLY_COLS = {
    "detailed_balance_efficiency_pct",
}

# Simulator-internal state / mechanism variables: the "why" behind observed
# behavior, not something any real monitoring system reports. Confirmed
# with the domain owner these are metadata about the generative process,
# never training inputs, even for the research/"full" tier — using them
# would let a model shortcut around learning the physical signal it's
# actually supposed to infer from indirect telemetry (e.g. aging_factor /
# soiling_factor / shading_factor directly CAUSE the power deviations a
# fault-detection model is meant to detect from power/irradiance/temp
# alone). vulnerability is additionally run-level (one Beta(2,5) draw per
# simulation run, not per-row) rather than a real per-timestep signal at
# all — the same failure mode as a leaked group/run identifier.
_SIMULATION_INTERNAL_COLS = {
    "soiling_factor", "aoi_correction_factor", "spectral_factor",
    "shading_factor", "inverter_efficiency", "inverter_loss_kw",
    "inverter_derating_factor", "snow_loss_factor", "mismatch_loss_factor",
    "dc_wiring_loss_factor", "ac_wiring_loss_factor", "availability_factor",
    "lid_factor", "bifacial_gain", "aging_factor",
    "vulnerability",
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
        Feature columns measurable by real monitoring hardware (plus the
        purely SCADA/timestamp-derived engineered features).
    iv_curve_features : list[str]
        Real IV-curve-tracer outputs (Voc, Jsc, FF, Vmpp, Jmpp,
        efficiency_pct) — not baseline SCADA, but achievable on advanced,
        smart-grid-connected installations that perform periodic IV
        curves. Research/"full"-tier only, never "deployment".
    stress_features : list[str]
        Weather-derived stress indicator columns (``stress_*``) —
        deterministic continuous intensity values computed from weather
        data (0 = no stress, 1.0 = at threshold, >1 = above threshold),
        not currently exposed by this pipeline's baseline SCADA feature
        set. Research-tier, included in "full" and "scada+stress".
    all_feature_cols : list[str]
        Union of the three feature lists above. Deliberately does NOT
        include simulator-internal columns (aging_factor, soiling_factor,
        vulnerability, etc.) or theoretical-only columns
        (detailed_balance_efficiency_pct) — see the module docstring.
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
            log.info("  Config: array=%.2f kWp, %d location(s) from batch_config.json",
                     self.array_kwp or 0, len(self.locations))
        else:
            log.info("  No batch_config.json — using auto-detection.")
            self.array_kwp = None
            self.locations = {}
            self.config    = {}
 
        # Fallback: discover locations directly from the parquet files'
        # own latitude/longitude columns whenever batch_config.json didn't
        # provide any — whether because it's absent entirely, or present
        # but structured differently than the sweep_summary.locations[]
        # schema above expects. array_kwp already has an equivalent
        # fallback (estimated from dc_power_kw in _discover_columns);
        # locations previously had none.
        if not self.locations:
            discovered = self._discover_locations_from_parquet()
            if discovered:
                self.locations = discovered
                log.info("  Discovered %d location(s) directly from parquet "
                         "latitude/longitude columns (batch_config.json "
                         "provided none).", len(self.locations))

    def _discover_locations_from_parquet(self) -> dict[tuple, str]:
        """Discover unique (lat, lon) pairs by reading them directly from
        every parquet file in the batch.

        Fallback source of truth for locations, used whenever
        ``batch_config.json`` is missing or doesn't yield anything under
        its expected schema. Reads only the two coordinate columns from
        one row of each file — cheap even across many files, since
        parquet's columnar layout means this never touches feature data.

        Returns
        -------
        dict[tuple[float, float], str]
            Same shape as the JSON-derived ``self.locations``: keys are
            ``(round(lat, 2), round(lon, 2))``. Values fall back to a
            ``"{lat}_{lon}"`` string since no human-readable place name is
            available from the parquet data alone.
        """
        files = sorted(self.data_dir.glob("*.parquet"))
        discovered: dict[tuple, str] = {}
        for f in files:
            try:
                row = pd.read_parquet(f, columns=["latitude", "longitude"]).iloc[0]
            except Exception:
                continue
            lat = round(float(row["latitude"]), 2)
            lon = round(float(row["longitude"]), 2)
            discovered.setdefault((lat, lon), f"{lat}_{lon}")
        return discovered

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

        non_feature = (
            self.meta_cols | self.label_cols | self.combo_cols
            | self.noise_cols | {"timestamp", "run_id"}
            | {c for c in all_cols if c.endswith("_true")}
            | _THEORETICAL_ONLY_COLS      # no real install can ever measure these
            | _SIMULATION_INTERNAL_COLS   # generative-process metadata, not signal
        )
        feature_candidates = all_cols - non_feature

        self.stress_features: list[str] = sorted(
            c for c in feature_candidates
            if c.startswith(_STRESS_PREFIX) and c not in constant_cols
        )
        self.scada_features: list[str] = sorted(
            c for c in feature_candidates
            if c in _SCADA_MEASURABLE and c not in constant_cols
        )
        self.iv_curve_features: list[str] = sorted(
            c for c in feature_candidates
            if c in _IV_CURVE_COLS and c not in constant_cols
        )
        self.all_feature_cols: list[str] = (
            self.scada_features + self.iv_curve_features + self.stress_features
        )

        # DEFAULT-DENY check: anything left in feature_candidates belongs to
        # none of the three recognised categories above, so it was NEVER
        # exposed as a feature — unlike the old design, where an unrecognised
        # column fell through into a catch-all "device_features" bucket and
        # was silently trained on (this is exactly how aging_factor,
        # soiling_factor, and vulnerability ended up as model inputs before).
        # Surfacing this list means a genuinely new, useful column in a
        # future SOLEY batch is at worst a visible warning to triage — not a
        # silent leak — and a genuinely internal one stays out without
        # anyone having to notice and hand-add it to the exclusion set above.
        categorized = (
            set(self.scada_features) | set(self.stress_features)
            | set(self.iv_curve_features)
        )
        uncategorized = feature_candidates - categorized - constant_cols
        if uncategorized:
            log.warning(
                "  %d column(s) matched none of the known feature categories "
                "and were EXCLUDED as a precaution (never silently trained "
                "on): %s. If any of these are real, usable signals, add them "
                "to _SCADA_MEASURABLE or _IV_CURVE_COLS in batch_config.py; "
                "if they're simulator-internal, add them to "
                "_SIMULATION_INTERNAL_COLS instead so this warning stops "
                "recurring.",
                len(uncategorized), sorted(uncategorized),
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

        theoretical_found = _THEORETICAL_ONLY_COLS & all_cols
        internal_found    = _SIMULATION_INTERNAL_COLS & all_cols
        if theoretical_found or internal_found:
            log.warning(
                "  Simulation-only columns found in data and excluded from "
                "all feature sets: %s (theoretical-only), %s "
                "(simulator-internal — never a training input, even for "
                "\"full\")",
                sorted(theoretical_found) or "none",
                sorted(internal_found) or "none",
            )
        log.info(
            "  Columns: %d SCADA, %d IV-curve, %d stress, "
            "%d constant (excluded), %d simulation-only (excluded), "
            "%d uncategorized (excluded), %d total features",
            len(self.scada_features), len(self.iv_curve_features),
            len(self.stress_features), len(constant_cols),
            len(theoretical_found) + len(internal_found), len(uncategorized),
            len(self.all_feature_cols),
        )

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------

    @property
    def device_features(self) -> list[str]:
        """Deprecated alias for :attr:`iv_curve_features`.

        The old ``device_features`` name referred to a broad catch-all
        (anything not SCADA-measurable or stress-prefixed) that turned out
        to silently include simulator-internal columns like
        ``aging_factor`` and ``vulnerability`` as trainable features — see
        the module docstring. That catch-all no longer exists; this
        property exists only so an old notebook cell or script referencing
        ``cfg.device_features`` degrades to a clear, actionable warning
        instead of an ``AttributeError``, rather than for anyone to keep
        using it — update callers to ``iv_curve_features`` instead.
        """
        log.warning(
            '  cfg.device_features is a deprecated alias for '
            'cfg.iv_curve_features (see its docstring) — update the '
            'caller; this alias may be removed in future.'
        )
        return self.iv_curve_features

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
        mode : {"full", "scada", "scada+stress", "scada+iv_curve"}
            "scada"          — real-SCADA-only + purely SCADA/timestamp-
                               derived engineered features. The deployment
                               tier: works on any real installation's
                               existing monitoring data, today.
            "scada+stress"   — adds the weather-stress indicators. These
                               ARE deterministically derivable from weather
                               data (not simulator-internal state the way
                               aging_factor etc. are), but this pipeline
                               doesn't compute them for real installations
                               today, so they stay an explicit, opt-in
                               research variant rather than part of
                               "scada"/deployment by default.
            "scada+iv_curve" — adds real IV-curve-tracer outputs (Voc, Jsc,
                               FF, Vmpp, Jmpp, efficiency_pct). Not baseline
                               SCADA, but a real capability on advanced,
                               smart-grid-connected installations that
                               perform periodic IV curves.
            "full"           — scada + stress + iv_curve: every category
                               that's real-world-*achievable* on some
                               installation, combined. The research/
                               ablation tier — NOT deployment-realistic on
                               a baseline installation. Never includes any
                               simulator-internal-only or theoretical-only
                               column (aging_factor, soiling_factor,
                               vulnerability, detailed_balance_efficiency_pct,
                               etc.) — those are excluded unconditionally,
                               for every mode, at column-discovery time.
        """
        eng = list(_ENGINEERED_FEATURES)
        scada = list(self.scada_features) + eng

        if mode == "scada":
            return scada
        if mode == "scada+stress":
            return scada + list(self.stress_features)
        if mode == "scada+iv_curve":
            return scada + list(self.iv_curve_features)
        if mode == "scada+device":
            log.warning(
                '  feature_set("scada+device") is a deprecated alias — the '
                'catch-all "device_features" category it used to mean has '
                'been split into iv_curve_features (real, achievable on '
                'advanced installs) and simulator-internal columns (always '
                'excluded, see _SIMULATION_INTERNAL_COLS). Returning '
                '"scada+iv_curve" instead.'
            )
            return scada + list(self.iv_curve_features)
        # full
        return scada + list(self.stress_features) + list(self.iv_curve_features)

    def __repr__(self) -> str:
        return (
            f"BatchConfig("
            f"array_kwp={self.array_kwp:.2f}, "
            f"locations={len(self.locations)}, "
            f"features={len(self.all_feature_cols)})"
        )
