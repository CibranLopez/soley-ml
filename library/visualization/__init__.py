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

plot_shap_summary(values, feature_values, feature_names, title, path, method)
    Horizontal bar chart of mean(|attribution|) per feature.
    Accepts SHAP values (tabular) or Captum attributions (sequence models)
    — both collapse to the same (n_samples, n_features) shape before this
    function is called, so a single renderer covers both cases.
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
    iv_curve_features: list[str] | None = None,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of RF feature importances, colour-coded by group.

    Colours:  blue = SCADA,  green = IV curve,  amber = stress.
    """
    importances = model.feature_importances_
    top_n       = min(top_n, len(feature_names))
    indices     = np.argsort(importances)[-top_n:]

    iv_curve_set = set(iv_curve_features or [])

    colors = []
    for i in indices:
        name = feature_names[i]
        if name.startswith("stress_"):
            colors.append(_AMBER)
        elif name in iv_curve_set:
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
        "(blue=SCADA  green=IV curve  amber=stress)",
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
#  Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation(ablation_results: list[dict], path: Path) -> None:
    """Bar chart of AUC across feature subsets.

    Defensively treats a missing/None AUC (``classification_metrics``
    stores ``None`` when AUC computation itself fails — e.g. a degenerate
    test fold with only one class present) as ``np.nan`` rather than
    letting it reach ``min()``/bar-height arithmetic un-guarded, which
    would raise ``TypeError``. This is defense-in-depth alongside the fix
    in ``unified.py``'s ``run_feature_ablation`` (which now substitutes
    ``np.nan`` before this function is ever called) — hardening the
    plotting function itself means it can't be broken again by some other,
    future caller making the same mistake.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = [r["feature_set"] for r in ablation_results]
    aucs   = [(r["auc"] if r["auc"] is not None else np.nan)
              for r in ablation_results]
    colors = [_BLUE, _AMBER, _GREEN, _PURPLE]

    bars = ax.bar(range(len(ablation_results)), aucs,
                  color=colors[:len(ablation_results)], width=0.6)
    ax.set_xticks(range(len(ablation_results)))
    ax.set_xticklabels(names, fontsize=10, rotation=5, ha="center")
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_title("Feature Ablation — SOLEY Physics vs. SCADA Only", fontsize=13)
    finite_aucs = [a for a in aucs if not np.isnan(a)]
    min_auc = min(finite_aucs) if finite_aucs else 0.0
    ax.set_ylim(max(0, min_auc - 0.05), 1.02)
    for bar, v, n in zip(bars, aucs, [r["n_features"] for r in ablation_results]):
        label = f"AUC={v:.4f}\n({n} feat.)" if not np.isnan(v) else f"AUC=N/A\n({n} feat.)"
        y_pos = (v if not np.isnan(v) else 0) + 0.005
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label, ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", Path(path).name)


# ---------------------------------------------------------------------------
#  Cross-location bar chart
# ---------------------------------------------------------------------------

def plot_cross_location(results: list[dict], path: Path) -> None:
    """Side-by-side detection / classification AUC bars per test site.

    Defensively treats a missing/None AUC as ``np.nan`` before plotting —
    see the docstring note in :func:`plot_ablation` for why this same
    guard is needed here. Without it, a single fold where AUC computation
    failed (e.g. a location whose test data happened to be degenerate for
    classification — too few classes present) would crash the entire
    chart via ``ax.bar()``'s internal arithmetic on a Python ``None``.
    """
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
        vals = [(r[key] if r[key] is not None else np.nan) for r in results]
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


# ---------------------------------------------------------------------------
#  SHAP / attribution summary  (Task C — tabular + sequence models)
# ---------------------------------------------------------------------------

def plot_shap_summary(
    values: np.ndarray,
    feature_values: np.ndarray | None,
    feature_names: list[str] | None,
    title: str,
    path: Path,
    method: str = "shap",
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of mean(|attribution|) per feature.

    A single renderer for both SHAP values (tabular models via
    ``shap.TreeExplainer`` / ``shap.LinearExplainer``) and gradient-based
    attributions (PyTorch sequence models via Captum
    ``IntegratedGradients``). Both sources collapse to the same
    ``(n_samples, n_features)`` shape before calling this function, so
    the plot logic is identical regardless of model family.

    Follows the same conventions as the rest of this module: matplotlib
    Agg backend, save to a ``Path``, ``log.info`` on completion.
    Complements (does not replace) ``plot_importance``: importance shows
    the model's global learned weights; this shows which features actually
    drove the model's predictions on the held-out test set.

    Parameters
    ----------
    values : np.ndarray, shape (n_samples, n_features)
        SHAP values or attribution scores. Already collapsed to 2-D:
        binary classifiers → positive/faulted class only;
        multi-class → mean(|.|) across classes;
        sequence models → mean over the time/window dimension.
    feature_values : np.ndarray, shape (n_samples, n_features) or None
        The underlying feature values the attributions correspond to.
        Currently unused in the bar-chart rendering but kept in the
        signature for future beeswarm-style dot plots.
    feature_names : list[str] or None
        If None, features are labelled f0, f1, …
    title : str
    path : Path
    method : str
        Attribution method label shown in the x-axis and plot title so
        readers know what they're looking at: ``"shap"`` (TreeExplainer /
        LinearExplainer), ``"integrated_gradients"`` (Captum IG), etc.
    top_n : int
        Number of top features (by mean |attribution|) to display.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        log.info(
            "  plot_shap_summary: expected 2-D array, got shape %s — "
            "skipping plot for %s", values.shape, Path(path).name
        )
        return

    n_features   = values.shape[1]
    names        = (feature_names if feature_names is not None
                    else [f"f{i}" for i in range(n_features)])
    mean_abs     = np.abs(values).mean(axis=0)   # (n_features,)
    top_n        = min(top_n, n_features)
    order        = np.argsort(mean_abs)[-top_n:]  # ascending → bottom of chart

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))

    # Colour-code by sign of the mean attribution so the chart is
    # informative even without a beeswarm: positive mean → blue (fault
    # indicator), negative mean → red (healthy indicator).
    mean_signed  = values.mean(axis=0)
    bar_colors   = [_BLUE if mean_signed[i] >= 0 else _RED for i in order]

    ax.barh(range(len(order)), mean_abs[order], color=bar_colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=10)
    ax.set_xlabel(f"mean(|{method} attribution|)", fontsize=11)
    ax.set_title(
        f"{title}\n"
        f"(blue = pushes toward fault, red = pushes toward healthy)",
        fontsize=12,
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)


__all__ = [
    "plot_confusion",
    "plot_importance",
    "plot_training_curves",
    "plot_per_fault_f1",
    "plot_ablation",
    "plot_cross_location",
    "plot_shap_summary",
]
