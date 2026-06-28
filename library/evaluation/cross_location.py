"""
library.evaluation.cross_location
=====================================

The RF-only ``run_cross_location`` that lived here has been removed.

The model-agnostic version in :mod:`library.evaluation.unified` supersedes
it: it accepts ``model_modes=`` to run any tabular or sequence model and
uses the same file-level registry splits as the main training pipeline.

``library.evaluation.run_cross_location`` still works unchanged — it now
always resolves to the unified version.
"""
