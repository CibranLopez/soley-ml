"""
library.evaluation.ablation
==============================

The RF-only ``run_feature_ablation`` that lived here has been removed.

The model-agnostic version in :mod:`library.evaluation.unified` supersedes
it: it accepts ``model_modes=`` to run any tabular or sequence model and
uses the same file-level registry splits as the main training pipeline.

``library.evaluation.run_feature_ablation`` still works unchanged — it now
always resolves to the unified version.
"""
