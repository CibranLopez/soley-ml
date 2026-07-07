"""
library.models
===============

Model zoo for solar PV fault detection and classification.

  definitions  — model builders and mode registries
  dataset      — WindowDataset and streaming memmap builder
  trainer      — unified tabular + sequence training/evaluation
  pipeline     — notebook-facing convenience wrapper
"""

from .pipeline import run_model_task
from .trainer import run_task
from .definitions import build_model, TABULAR_ESTIMATORS, SEQUENCE_MODES

__all__ = [
    "run_model_task",
    "run_task",
    "build_model",
    "TABULAR_ESTIMATORS",
    "SEQUENCE_MODES",
]
