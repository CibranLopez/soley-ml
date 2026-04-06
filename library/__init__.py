"""
library — Solar PV Fault Detection & Classification Library
=============================================================

Modules
-------
library.config        — BatchConfig: auto-detects array capacity, locations,
                         and feature columns from any SOLEY batch output.
library.data          — Data loading (RF flat + PyTorch streaming).
library.features      — Feature engineering (PR, time encoding, etc.).
library.models        — Random Forest, MLP, LSTM, Hybrid model zoo.
library.evaluation    — Metrics, CV, temporal split, cross-location, ablation.
library.visualization — Confusion matrices, importance bars, training curves.
library.utils         — File registry, label encoding, logging helpers.
"""

__version__ = "1.0.0"
