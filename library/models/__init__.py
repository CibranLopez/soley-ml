"""
library.models
===============

Model zoo for solar PV fault detection and classification.

  random_forest   — scikit-learn RandomForestClassifier wrappers
  neural_network  — PyTorch MLP / LSTM / Hybrid architecture
  dataset         — WindowDataset and streaming memmap builder
  trainer         — PyTorch training loop and evaluator
"""

from .pipeline import run_model_task
from .random_forest import (
  run_rf_task,
  train_rf_classification,
  train_rf_detection,
  train_rf_from_registry,
)
from .neural_network import build_model

__all__ = [
  "run_model_task",
  "run_rf_task",
    "train_rf_detection",
    "train_rf_classification",
    "train_rf_from_registry",
    "build_model",
]
