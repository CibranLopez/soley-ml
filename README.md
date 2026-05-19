# soley-ml

Python package for unified solar PV fault detection and classification workflows.

## What is included

- `library.config`: batch-aware configuration and feature set handling.
- `library.data`: registry and split utilities plus data loading helpers.
- `library.models`: neural and random-forest training pipelines.
- `library.evaluation`: temporal, cross-location, and ablation evaluations.
- `library.visualization`: plots for metrics, confusion matrices, and curves.
- `library.utils`: logging and runtime environment helpers.

## Installation

From the repository root:

```bash
pip install .
```

For editable development installs:

```bash
pip install -e .
```

## Quick import check

```python
import library
print(library.__version__)
```

## Notes

- Jupyter notebooks in this repository (`01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb`) are examples and are not installed as package modules.
- Model artifacts and data folders are intentionally excluded from package discovery.
