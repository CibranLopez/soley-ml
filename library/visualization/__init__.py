"""
library.visualization
=====================

Publication-quality plots for fault detection and classification results.

plot_confusion(y_true, y_pred, class_names, title, path)
    Annotated confusion matrix (counts in each cell).

plot_importance(model, feature_names, title, path, ...)
    Horizontal bar chart of RF feature importances, colour-coded by group.

plot_training_curves(history, title, path)
    Loss and accuracy curves for PyTorch training.

plot_per_fault_f1(reports_by_model, class_names, path)
    Grouped bar chart comparing F1 score per fault type across model variants.

plot_ablation(ablation_results, path)
    Bar chart comparing AUC across feature subsets.

plot_cross_location(results, path)
    Side-by-side bar charts for detection and classification AUC per site.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("library")

# Colour palette (consistent across all plots)
_BLUE   = "#2563eb"
_GREEN  = "#10b981"
_AMBER  = "#f59e0b"
_PURPLE = "#8b5cf6"
_RED    = "#dc2626"

_MODEL_COLORS = {
    "mlp": _BLUE,
    "lstm": _AMBER,
    "hybrid": _GREEN,
    "random_forest": _PURPLE,
}


# ---------------------------------------------------------------------------
#  Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion(
    y_true,
    y_pred,
    class_names: list[str],
    title: str,
    path: Path,
) -> None:
    """Save an annotated confusion matrix PNG.

    Parameters
    ----------
    y_true, y_pred : array-like
        Integer class indices or string labels.
    class_names : list[str]
    title : str
    path : Path
    """
    from sklearn.metrics import confusion_matrix

    y_true = list(y_true)
    y_pred = list(y_pred)

    unique = sorted(set(y_true) | set(y_pred),
                    key=lambda v: (class_names.index(v)
                                   if isinstance(v, str) and v in class_names
                                   else v))
    cm = confusion_matrix(y_true, y_pred, labels=unique)
    n  = len(unique)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 0.9)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color=color, fontsize=9 if n > 6 else 11)

    labels = [class_names[unique.index(u)] if u in class_names else str(u)
              for u in unique]
    ax.set_xticks(range(n));  ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n));  ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"{title} — Confusion Matrix", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Feature importance
# ---------------------------------------------------------------------------

def plot_importance(
    model,
    feature_names: list[str],
    title: str,
    path: Path,
    scada_features: list[str] | None = None,
    device_features: list[str] | None = None,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of RF feature importances, colour-coded by group.

    Colours:  blue = SCADA,  green = device physics,  amber = stress.
    """
    importances = model.feature_importances_
    top_n       = min(top_n, len(feature_names))
    indices     = np.argsort(importances)[-top_n:]

    scada_set  = set(scada_features  or [])
    device_set = set(device_features or [])

    colors = []
    for i in indices:
        name = feature_names[i]
        if name.startswith("stress_"):
            colors.append(_AMBER)
        elif name in device_set:
            colors.append(_GREEN)
        else:
            colors.append(_BLUE)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(
        f"{title} — Top {top_n} Features\n"
        "(blue=SCADA  green=device physics  amber=stress)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    title: str,
    path: Path,
) -> None:
    """Loss and accuracy curves for PyTorch training.

    Parameters
    ----------
    history : dict  keys: train_loss, val_loss, val_acc
    title : str
    path : Path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train",      color=_BLUE)
    ax1.plot(epochs, history["val_loss"],   label="Validation", color=_RED)
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend();  ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="Validation", color=_GREEN)
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend();  ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Per-fault F1 comparison
# ---------------------------------------------------------------------------

def plot_per_fault_f1(
    reports_by_model: dict,
    class_names: list[str],
    path: Path,
) -> None:
    """Grouped bar chart: F1 per fault type across model variants.

    Parameters
    ----------
    reports_by_model : dict  ``{mode_name: classification_report_dict}``
    class_names : list[str]
    path : Path
    """
    models   = list(reports_by_model.keys())
    n_faults = len(class_names)
    x        = np.arange(n_faults)
    width    = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(10, n_faults * 1.2), 5))

    for i, m in enumerate(models):
        report    = reports_by_model[m]
        f1_scores = [report.get(cls, {}).get("f1-score", 0.0)
                     for cls in class_names]
        ax.bar(x + i * width, f1_scores, width,
               label=m.upper(),
               color=_MODEL_COLORS.get(m, "#666666"))

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Fault-Type F1 Score by Model Architecture", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Ablation bar chart  (standalone — evaluation/ablation.py also draws one)
# ---------------------------------------------------------------------------

def plot_ablation(ablation_results: list[dict], path: Path) -> None:
    """Bar chart of AUC across feature subsets."""
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = [r["feature_set"] for r in ablation_results]
    aucs   = [r["auc"]         for r in ablation_results]
    colors = [_BLUE, _AMBER, _GREEN, _PURPLE]

    bars = ax.bar(range(len(ablation_results)), aucs,
                  color=colors[:len(ablation_results)], width=0.6)
    ax.set_xticks(range(len(ablation_results)))
    ax.set_xticklabels(names, fontsize=10, rotation=5, ha="center")
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_title("Feature Ablation — SOLEY Physics vs. SCADA Only", fontsize=13)
    min_auc = min(aucs)
    ax.set_ylim(max(0, min_auc - 0.05), 1.02)
    for bar, v, n in zip(bars, aucs, [r["n_features"] for r in ablation_results]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"AUC={v:.4f}\n({n} feat.)",
                ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Cross-location bar chart  (standalone version)
# ---------------------------------------------------------------------------

def plot_cross_location(results: list[dict], path: Path) -> None:
    """Side-by-side detection / classification AUC bars per test site."""
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["test_location"] for r in results]
    x     = range(len(results))

    for ax, key, color, title in [
        (axes[0], "detection_auc",      _BLUE,
         "Cross-Location Detection"),
        (axes[1], "classification_auc", _GREEN,
         "Cross-Location Classification"),
    ]:
        vals = [r[key] for r in results]
        bars = ax.bar(x, vals, color=color, width=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("ROC AUC", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random")
        ax.legend(fontsize=10)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.3f}", ha="center", fontsize=11,
                        fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


__all__ = [
    "plot_confusion",
    "plot_importance",
    "plot_training_curves",
    "plot_per_fault_f1",
    "plot_ablation",
    "plot_cross_location",
]
