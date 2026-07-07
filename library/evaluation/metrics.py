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

    # Explicit labels=range(len(class_names)) is required, not optional:
    # without it, classification_report infers its label set from whatever
    # is actually present in y_true/y_pred for THIS call. If a specific
    # fold's test data doesn't cover every class the shared LabelEncoder
    # knows about — plausible whenever this is called per-site (cross-
    # location) or per-year (temporal split), where one site/year may
    # simply not contain every fault type — that inferred set is smaller
    # than class_names, and classification_report raises an uncaught
    # ValueError ("Number of classes, N, does not match size of
    # target_names, M"), crashing the whole run_task call rather than
    # just this one metric. Passing labels= explicitly makes the label set
    # always the full vocabulary; a class absent from this fold correctly
    # shows up as a zero-support row (zero_division=0 handles the
    # resulting 0/0 precision/recall) instead of raising.
    labels = range(len(class_names)) if class_names is not None else None

    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    log.info("")
    if task:
        log.info("  %s", task)
    log.info(
        classification_report(y_true, y_pred,
                              labels=labels,
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
