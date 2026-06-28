"""
library.evaluation.temporal
=============================

The RF-only ``run_temporal_split`` that lived here has been removed.

The model-agnostic version in :mod:`library.evaluation.unified` supersedes
it: it accepts ``model_modes=`` to run any tabular or sequence model, uses
the same file-level registry splits as the main training pipeline, and now
correctly handles multi-year parquet files by reading all unique ``sim_year``
values per file rather than assuming ``iloc[0]`` is representative.

``library.evaluation.run_temporal_split`` still works unchanged — it now
always resolves to the unified version.
"""
