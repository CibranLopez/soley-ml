"""
library.evaluation.metrics
============================

Thin wrappers around sklearn metrics that log results consistently
and return structured dicts for downstream reporting / plotting.
"""

import logging

import numpy as np

log = logging.getLogger("library")


def classification_metrics(
    y_true,
    y_pred,
    y_prob,
    class_names: list[str] | None = None,
    task: str = "",
) -> dict:
    """Compute and log classification metrics.

    Parameters
    ----------
    y_true, y_pred : array-like of int labels
    y_prob : array-like, shape (n_samples, n_classes)
    class_names : list[str] or None
    task : str  — label for log messages

    Returns
    -------
    dict with keys: auc, accuracy, report (sklearn dict)
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    log.info("")
    if task:
        log.info("  %s", task)
    log.info(
        classification_report(y_true, y_pred,
                              target_names=class_names, zero_division=0)
    )

    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob,
                                multi_class="ovr", average="weighted")
        log.info("  ROC AUC: %.4f", auc)
    except Exception:
        auc = None

    return {"auc": auc, "accuracy": acc, "report": rep}
