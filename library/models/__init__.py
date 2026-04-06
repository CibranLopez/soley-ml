"""
library.models
===============

Model zoo for solar PV fault detection and classification.

  random_forest   — scikit-learn RandomForestClassifier wrappers
  neural_network  — PyTorch MLP / LSTM / Hybrid architecture
  dataset         — WindowDataset and streaming memmap builder
  trainer         — PyTorch training loop and evaluator
"""

from .random_forest import train_rf_detection, train_rf_classification
from .neural_network import build_model

__all__ = [
    "train_rf_detection",
    "train_rf_classification",
    "build_model",
]
