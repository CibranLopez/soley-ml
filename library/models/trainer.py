"""
library.models.trainer
========================

Single training orchestrator for every model family this library
supports — tabular (Random Forest and any future scikit-learn-API
estimator) and sequence (MLP / LSTM / Hybrid PyTorch architectures).

Why one file, not one per family
----------------------------------
Until this revision, tabular and sequence models were trained by two
independent code paths (``random_forest.py`` and this file's old
``run_pytorch_task``) that each re-implemented: resolving train/val/test
entries from the registry, fitting a LabelEncoder for classification
tasks, and saving/returning artifacts in a slightly different shape. The
duplication was more than cosmetic — the two LabelEncoders were fit on
different inputs (tabular: actual ``fault_type`` values from training
*rows*; sequence: ``fault_type`` parsed from training *filenames*), which
could silently disagree on class-to-index ordering whenever a file's
filename-derived type didn't perfectly match what its rows contained.

:func:`run_task` is now the *only* entry point. It performs every piece of
setup that must be identical across model families exactly once — entry
resolution, fault-type discovery from real data, LabelEncoder fitting —
then hands the shared, already-consistent state to whichever per-family
helper(s) the requested ``model_modes`` need:

* :func:`_run_tabular_modes` — loads the train/test DataFrame once via
  :func:`~library.data.loader.load_split`, applies one shared stratified
  row-level subsample, then fits every requested tabular estimator
  (:data:`~library.models.definitions.TABULAR_ESTIMATORS`) against the
  identical in-memory ``(X, y)`` arrays.
* :func:`_run_sequence_modes` — fits one shared ``StandardScaler`` and
  builds one shared set of memmap window arrays, then trains every
  requested sequence architecture (:data:`~library.models.definitions.SEQUENCE_MODES`)
  against the identical windows.

Both helpers return artifact dicts with the same key set (``model``,
``report``, ``auc``, ``label_encoder``, ``class_names``, ``feature_names``,
``y_true``, ``y_pred``, ``model_path``), so downstream code (manifests,
``library.evaluation.unified``) doesn't need to special-case which family
produced a given mode's result.

A note on "identical data" across families
---------------------------------------------
Every model family trained by one ``run_task`` call shares the same
*files* (the registry's train/val/test split is resolved once, up front)
and, for classification, the same label encoding. Within the tabular
family, every requested estimator additionally shares the exact same
*rows* (the stratified subsample is drawn once, not per estimator).
Sequence models cannot share row-level identity with the tabular family
by construction — they consume contiguous per-file windows via memmap,
not an arbitrarily-subsampled flat table — so "same files, same labels,
full data" is the strongest correctness guarantee available across the
tabular/sequence boundary; "same individual rows" only applies within the
tabular family, where the data representation actually permits it.

Module-level utilities (unchanged, model-family-agnostic)
------------------------------------------------------------
train_model(model, train_loader, val_loader, device, ...)
    PyTorch training loop: weighted cross-entropy, Adam + ReduceLROnPlateau,
    gradient clipping, early stopping.
evaluate_model(model, test_loader, device, class_names)
    Predictions, probabilities, classification report, ROC-AUC.
compute_class_weights(runs, n_classes)
    Inverse-frequency class weights from memmap label files.
plot_attribution(model, test_loader, ...)
    Opt-in gradient-based explainability (Captum IntegratedGradients) for
    sequence models, called only when ``compute_shap=True`` — see its own
    docstring for why it isn't named ``_maybe_...`` despite degrading
    gracefully internally.
"""

import logging

import numpy as np

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Class-weight computation (memmap-aware) — shared by every sequence mode
# ---------------------------------------------------------------------------

def compute_class_weights(runs: list[tuple], n_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights from memmap label files.

    Parameters
    ----------
    runs : list[tuple]
        ``(feat_path, feat_shape, lab_path, lab_shape, run_id)``
    n_classes : int

    Returns
    -------
    class_weights : np.ndarray, shape (n_classes,)
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    total  = 0
    for _fp, _fs, lp, ls, _ in runs:
        lab     = np.memmap(lp, dtype=np.int64, mode="r", shape=ls)
        c       = np.bincount(np.asarray(lab), minlength=n_classes)
        counts += c.astype(np.float64)
        total  += len(lab)
        del lab
    counts = np.maximum(counts, 1.0)
    return total / (n_classes * counts)


# ---------------------------------------------------------------------------
#  Training loop (PyTorch, shared by every sequence mode)
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    n_classes: int,
    class_weights: np.ndarray | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 7,
):
    """Train a PyTorch model with early stopping.

    Parameters
    ----------
    model : nn.Module
        Created by :func:`~library.models.definitions.build_model`.
    train_loader, val_loader : DataLoader
    device : torch.device
    n_classes : int
    class_weights : np.ndarray or None
        If provided, used for weighted cross-entropy.
    epochs : int
        Maximum training epochs.
    lr : float
        Initial learning rate for Adam.
    patience : int
        Early-stopping patience (in epochs without val-loss improvement).

    Returns
    -------
    model : nn.Module  (best checkpoint restored)
    history : dict
        Keys: ``train_loss``, ``val_loss``, ``val_acc``
    """
    import torch
    import torch.nn as nn

    model = model.to(device)

    if class_weights is not None:
        w         = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    history    = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val   = float("inf")
    best_state = None
    wait       = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        val_loss  = 0.0
        n_correct = 0
        n_total   = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(x_batch)
                loss    = criterion(logits, y_batch)
                val_loss += loss.item()
                preds     = logits.argmax(dim=1)
                n_correct += (preds == y_batch).sum().item()
                n_total   += len(y_batch)
        val_loss = val_loss / max(len(val_loader), 1)
        val_acc  = n_correct / max(n_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        log.info(
            "  Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

        # ---- Early stopping ----
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("  Early stopping at epoch %d (patience=%d)",
                         epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, history


# ---------------------------------------------------------------------------
#  Evaluation (PyTorch, shared by every sequence mode)
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    test_loader,
    device,
    class_names: list[str] | None = None,
):
    """Evaluate a trained model on a test DataLoader.

    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    device : torch.device
    class_names : list[str] or None

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_prob : np.ndarray  shape (n_samples, n_classes)
    report : dict         sklearn classification_report output_dict=True
    auc : float or None  ROC-AUC (binary: standard; multi-class: OvR weighted)
    """
    import torch

    from library.evaluation.metrics import classification_metrics

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits  = model(x_batch)
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_probs.append(probs)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    metrics = classification_metrics(
        y_true, y_pred, y_prob, class_names=class_names,
    )
    report = metrics["report"]
    auc    = metrics["auc"]

    return y_true, y_pred, y_prob, report, auc


# ---------------------------------------------------------------------------
#  Gradient-based attribution — opt-in explainability for sequence models
# ---------------------------------------------------------------------------

def plot_attribution(
    model,
    test_loader,
    feature_names: list[str],
    title: str,
    path,
    device,
    max_samples: int = 256,
) -> None:
    """Gradient-based attribution plot for a sequence model (mlp/lstm/hybrid).

    Renamed from the original ``_maybe_plot_attribution``: the "maybe" was
    redundant with how this function is already called. The *caller*
    decides whether to attempt attribution at all via the explicit
    ``compute_shap=True`` opt-in flag passed to :func:`run_task` — by the
    time this function is actually invoked, there is no more "maybe"
    about whether to try. What the old name was really describing is
    ordinary defensive error handling (missing optional dependency, no
    test data, a failed attribution call) — every function that touches
    optional dependencies or external data has to handle that, and we
    don't name those ``_maybe_read_file`` either. The hedge belongs in the
    function *body* (the try/except blocks below, which are unchanged),
    not in its name.

    ``shap.DeepExplainer`` can be unreliable with recurrent layers
    depending on the installed SHAP/PyTorch version combination (its
    hooks into LSTM internals are fragile across versions). We use
    Captum's ``IntegratedGradients`` instead: it's maintained directly by
    the PyTorch team, works against arbitrary ``nn.Module`` graphs with no
    special-casing for LSTM, and its attributions are axiomatically
    grounded (completeness: attributions sum to the prediction delta from
    a zero baseline) rather than approximated via sampling like
    ``GradientExplainer``.

    Scoped to a representative sample of windows (``max_samples``), not
    the full test set — integrated gradients needs multiple forward+
    backward passes per sample (``n_steps``), so it isn't cheap.

    Attributions are averaged over the time/window dimension before
    plotting, collapsing ``(n_samples, window, n_features)`` down to
    ``(n_samples, n_features)`` — the same shape a tabular SHAP path would
    produce — so a single ``plot_shap_summary`` can render either.

    Degrades gracefully (logs and returns) if ``captum`` isn't installed,
    if there's no test data, or if attribution computation fails, rather
    than failing the training run — this is opt-in instrumentation and a
    visualization failure shouldn't take a trained model down with it.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        log.info("  (captum not installed — skipping gradient attribution; "
                 "`pip install captum` to enable)")
        return

    import torch

    model.eval()
    xs, ys, n_collected = [], [], 0
    for x_batch, y_batch in test_loader:
        xs.append(x_batch)
        ys.append(y_batch)
        n_collected += x_batch.shape[0]
        if n_collected >= max_samples:
            break
    if not xs:
        log.info("  (no test windows available — skipping attribution)")
        return

    x_sample = torch.cat(xs, dim=0)[:max_samples].to(device)
    y_sample = torch.cat(ys, dim=0)[:max_samples].to(device)
    baseline = torch.zeros_like(x_sample)

    ig = IntegratedGradients(model)
    try:
        attributions = ig.attribute(
            x_sample, baselines=baseline, target=y_sample, n_steps=32,
        )
    except Exception as exc:
        log.info("  Integrated-gradients attribution failed (%s) — skipping.", exc)
        return

    # (n_samples, window, n_features) -> (n_samples, n_features)
    attr_by_feature = attributions.detach().cpu().numpy().mean(axis=1)
    x_by_feature    = x_sample.detach().cpu().numpy().mean(axis=1)

    try:
        from library.visualization import plot_shap_summary
    except ImportError:
        log.info("  (library.visualization.plot_shap_summary not available "
                 "yet — skipping attribution plot; computed attributions "
                 "but had nowhere to render them)")
        return

    plot_shap_summary(attr_by_feature, x_by_feature, feature_names, title, path,
                      method="integrated_gradients")


def _decode_labels(y_true, y_pred, le, is_cls: bool):
    """Decode integer labels back to their original string form for display
    and plotting, when this is a classification task with a fitted encoder.

    Both the tabular and sequence training paths need "the true/predicted
    labels, in human-readable form if this is classification" for their
    confusion-matrix plots and returned artifacts — previously each path
    carried its own identical copy of this if/else.

    Parameters
    ----------
    y_true, y_pred : array-like of int (classification) or 0/1 (detection)
    le : fitted LabelEncoder or None
    is_cls : bool

    Returns
    -------
    y_true_out, y_pred_out
        ``le.inverse_transform(...)`` output for classification;
        ``y_true``/``y_pred`` unchanged for detection.
    """
    if is_cls:
        return le.inverse_transform(y_true), le.inverse_transform(y_pred)
    return y_true, y_pred


# ---------------------------------------------------------------------------
#  Tabular family — one shared load + subsample, then one fit per estimator
# ---------------------------------------------------------------------------

def _fit_eval_save_tabular(
    mode: str,
    task_name: str,
    X_tr, y_tr, X_te, y_te,
    feature_names: list[str],
    is_cls: bool,
    le,
    class_names: list[str],
    output_dir,
    tag_base: str,
    cfg,
    save_deployment: bool = True,
    **estimator_kwargs,
) -> dict | None:
    """Fit, evaluate, plot, and save one tabular estimator.

    Returns ``None`` (and logs why) if the estimator's factory needs an
    optional dependency that isn't installed — this lets a multi-mode
    ``run_task`` call keep going for every other requested mode instead of
    crashing the whole run over one missing package.
    """
    import time

    from library.evaluation.metrics import classification_metrics
    from library.models.definitions import build_model
    from library.visualization import plot_confusion, plot_importance

    try:
        model = build_model(mode, **estimator_kwargs)
    except ImportError as exc:
        log.info("  Skipping tabular mode %r — %s", mode, exc)
        return None

    log.info("")
    log.info("  — Model: %s —", mode.upper())
    t0 = time.time()
    model.fit(X_tr, y_tr)
    log.info("  Trained in %.1fs", time.time() - t0)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    y_true_out, y_pred_out = _decode_labels(y_te, y_pred, le, is_cls)

    metrics = classification_metrics(
        y_te, y_pred, y_prob, class_names=class_names,
        task=f"{task_name} — {mode.upper()}",
    )
    auc, report = metrics["auc"], metrics["report"]

    tag = f"{tag_base}_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_true_out, y_pred_out, class_names,
                   f"{task_name} — {mode.upper()}",
                   output_dir / f"confusion_{tag}.png")

    if hasattr(model, "feature_importances_"):
        plot_importance(model, feature_names, f"{task_name} — {mode.upper()}",
                        output_dir / f"importance_{tag}.png",
                        scada_features=cfg.scada_features,
                        device_features=cfg.device_features)

    import joblib
    model_path = output_dir / f"model_{tag}.pkl"
    joblib.dump({
        "model":         model,
        "feature_names": feature_names,
        "class_names":   class_names,
        "label_encoder": le,
    }, model_path)
    log.info("  Saved: %s", model_path.name)

    # Deployment variant: same estimator family, SCADA + stress features
    # only (works on real SCADA data — no SOLEY-simulation-only physics).
    # Generalised from the original RF-only logic: works for whichever
    # `mode` was requested, since it goes through the same build_model().
    # Uses cfg.feature_set("scada+stress") — the same authoritative method
    # the notebooks use to build feature lists — rather than a hand-rolled
    # copy of the engineered-feature names, so this can't drift out of
    # sync with BatchConfig if a new engineered feature is ever added.
    if save_deployment:
        deploy_set = set(cfg.feature_set("scada+stress"))
        deploy_idx = [i for i, f in enumerate(feature_names) if f in deploy_set]
        if deploy_idx:
            deploy_feats = [feature_names[i] for i in deploy_idx]
            try:
                dep_model = build_model(mode, **estimator_kwargs)
            except ImportError:
                dep_model = None
            if dep_model is not None:
                log.info("  Training deployment model (SCADA+stress only) …")
                dep_model.fit(X_tr[:, deploy_idx], y_tr)
                # Reuse the same classification_metrics() helper the primary
                # model uses above, rather than re-deriving AUC inline. The
                # two must agree on binary-vs-multiclass handling — that
                # helper branches on the *actual* y_prob column count, which
                # is the correct signal (a classification task can still
                # have exactly 2 classes in a given split, in which case the
                # binary path is right even though is_cls=True for the task
                # as a whole).
                try:
                    dep_prob = dep_model.predict_proba(X_te[:, deploy_idx])
                    dep_metrics = classification_metrics(
                        y_te, dep_model.predict(X_te[:, deploy_idx]), dep_prob,
                        class_names=class_names,
                        task=f"{task_name} — {mode.upper()} — Deployment",
                    )
                    if dep_metrics["auc"] is not None:
                        log.info("  Deployment AUC: %.4f", dep_metrics["auc"])
                except Exception:
                    pass
                dep_path = output_dir / f"deployment_{tag}.pkl"
                joblib.dump({
                    "model":         dep_model,
                    "feature_names": deploy_feats,
                    "class_names":   class_names,
                    "label_encoder": le,
                    "note": "Deployment model: SCADA + stress features only.",
                }, dep_path)
                log.info("  Saved deployment model: %s", dep_path.name)

    return {
        "model":         model,
        "report":        report,
        "auc":           auc,
        "label_encoder": le,
        "class_names":   class_names,
        "feature_names": feature_names,
        "y_true":        y_true_out,
        "y_pred":        y_pred_out,
        "model_path":    str(model_path),
    }


def _run_tabular_modes(
    modes: list[str],
    task_name: str,
    target_col: str,
    is_cls: bool,
    train_entries: list[dict],
    val_entries: list[dict],
    test_entries: list[dict],
    feature_cols: list[str],
    cfg,
    le,
    class_names: list[str],
    output_dir,
    artifact_prefix: str | None,
    lookback_window: int,
    max_train_rows: int | None,
    random_state: int,
    tabular_kwargs: dict[str, dict] | None,
    save_deployment: bool,
    return_details: bool,
) -> dict:
    """Fit every requested tabular estimator against one shared data load.

    Loads train/test exactly once for the whole call (not once per mode),
    applies the file-level fault-active filter and the shared, stratified
    row-level subsample exactly once, then loops over ``modes`` reusing
    the identical ``(X, y)`` arrays for every estimator. This is what
    guarantees that — for example — a second tabular mode added alongside
    ``"random_forest"`` in the same call trains on literally the same
    rows, not an independently-drawn subsample (the previous design drew
    a fresh ``random_state``-seeded sample inside the single RF-only
    trainer; with only one tabular mode that was harmless, but it wasn't
    structurally guaranteed to stay that way).
    """
    from library.data.loader import load_split, stratified_subsample
    from library.features import build_feature_matrix

    log.info("")
    log.info("=" * 65)
    log.info("  TABULAR %s  —  modes: %s", task_name.upper(), ", ".join(modes))
    log.info("=" * 65)

    log.info("  Loading train files …")
    train_df = load_split(train_entries, cfg, lookback_window=lookback_window)
    log.info("  Loading test  files …")
    test_df  = load_split(test_entries,  cfg, lookback_window=lookback_window)

    if is_cls:
        if "fault_active" in train_df.columns:
            train_df = train_df[train_df["fault_active"]].copy()
        if "fault_active" in test_df.columns:
            test_df = test_df[test_df["fault_active"]].copy()
        if not train_df.empty and not test_df.empty:
            # The shared LabelEncoder was fit on train+val data; a fault
            # type confined entirely to test files wouldn't be in its
            # vocabulary. Drop those rows rather than crash on transform().
            known = test_df["fault_type"].isin(le.classes_)
            n_unknown = int((~known).sum())
            if n_unknown:
                log.info("  Dropping %d test rows with fault types unseen "
                         "in the shared label encoder", n_unknown)
                test_df = test_df.loc[known].copy()

    if train_df.empty or test_df.empty:
        log.info("  Not enough data — skipping tabular modes.")
        return {}

    # Shared, stratified subsample — applied ONCE so every tabular mode in
    # this call trains on the identical set of rows. (Also stratified, not
    # a plain random sample: with class sizes spanning roughly an order of
    # magnitude — e.g. curtailment vs. pid — a plain sample could eliminate
    # the rarest classes entirely once the cap is below the full dataset.)
    # Uses the same stratified_subsample() helper load_split() itself uses,
    # rather than a second, independently-maintained copy of the same
    # try/except-ValueError-fallback logic.
    if max_train_rows and len(train_df) > max_train_rows:
        strat_col = "fault_type" if is_cls else "fault_active"
        train_df = stratified_subsample(train_df, max_train_rows, strat_col,
                                        random_state)

    X_tr, used = build_feature_matrix(train_df, feature_cols)
    X_te, _    = build_feature_matrix(test_df,  used)

    if is_cls:
        y_tr = le.transform(train_df["fault_type"].values)
        y_te = le.transform(test_df["fault_type"].values)
    else:
        y_tr = train_df["fault_active"].values.astype(int)
        y_te = test_df["fault_active"].values.astype(int)

    log.info("  Train: %s rows  Test: %s rows  Features: %d",
             f"{len(X_tr):,}", f"{len(X_te):,}", len(used))

    tabular_kwargs = tabular_kwargs or {}
    tag_base = artifact_prefix or task_name.lower().replace(" ", "_")
    results: dict = {}

    for mode in modes:
        kwargs = dict(tabular_kwargs.get(mode, {}))
        kwargs.setdefault("random_state", random_state)
        details = _fit_eval_save_tabular(
            mode, task_name, X_tr, y_tr, X_te, y_te, used,
            is_cls, le, class_names, output_dir, tag_base, cfg,
            save_deployment=save_deployment, **kwargs,
        )
        if details:
            results[mode] = details if return_details else details["report"]

    return results


# ---------------------------------------------------------------------------
#  Sequence family — one shared scaler + memmap build, then one fit per mode
# ---------------------------------------------------------------------------

def _run_sequence_modes(
    modes: list[str],
    task_name: str,
    target_col: str,
    is_cls: bool,
    train_entries: list[dict],
    val_entries: list[dict],
    test_entries: list[dict],
    feature_cols: list[str],
    cfg,
    le,
    class_names: list[str],
    output_dir,
    artifact_prefix: str | None,
    window_size: int,
    stride: int,
    batch_size: int,
    epochs: int,
    device,
    patience: int,
    num_workers: int,
    lookback_window: int,
    compute_shap: bool,
    shap_sample_size: int,
    random_state: int,
    return_details: bool,
) -> dict:
    """Fit every requested sequence architecture against one shared scaler
    and memmap build (identical windows for every mode in this call — this
    sharing already existed before the unification; what changed is that
    this function now *receives* its LabelEncoder from :func:`run_task`
    instead of fitting its own, guaranteeing it agrees with whatever the
    tabular family in the same call used).

    ``random_state`` seeds ``torch``/``numpy``/``random`` immediately before
    building and training each mode. Previously only the tabular family had
    any reproducibility guarantee (via sklearn's own ``random_state``); the
    sequence family had none — weight initialisation and DataLoader
    shuffling were both unseeded, so training ``mlp`` twice with identical
    data and hyperparameters produced different metrics each run. Seeding
    per-mode (not once for the whole function) also makes each mode's
    result independent of what order it was trained in within the same
    call.
    """
    import shutil
    import random
    from pathlib import Path

    import joblib
    import torch
    from torch.utils.data import DataLoader

    from library.models.dataset import (
        WindowDataset,
        build_memmap_arrays,
        fit_scaler_streaming,
    )
    from library.models.definitions import build_model
    from library.visualization import plot_confusion, plot_training_curves

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = artifact_prefix or task_name.lower().replace(" ", "_")

    log.info("")
    log.info("=" * 65)
    log.info("  SEQUENCE %s  —  modes: %s", task_name.upper(), ", ".join(modes))
    log.info("=" * 65)

    n_classes = len(class_names)
    log.info("  Classes: %d  Files: train=%d, val=%d, test=%d",
             n_classes, len(train_entries), len(val_entries), len(test_entries))

    log.info("  Fitting scaler …")
    scaler, available = fit_scaler_streaming(
        train_entries, feature_cols, cfg, lookback_window=lookback_window)
    n_features = len(available)

    log.info("  Building train arrays (%d files) …", len(train_entries))
    train_runs, cache_train = build_memmap_arrays(
        train_entries, available, target_col, cfg, scaler, le, is_cls,
        lookback_window=lookback_window)
    log.info("  Building val arrays (%d files) …", len(val_entries))
    val_runs, cache_val = build_memmap_arrays(
        val_entries, available, target_col, cfg, scaler, le, is_cls,
        lookback_window=lookback_window)
    log.info("  Building test arrays (%d files) …", len(test_entries))
    test_runs, cache_test = build_memmap_arrays(
        test_entries, available, target_col, cfg, scaler, le, is_cls,
        lookback_window=lookback_window)

    joblib.dump(scaler, output_dir / f"scaler_{tag}.pkl")
    if le is not None:
        joblib.dump(le, output_dir / f"label_encoder_{tag}.pkl")
    log.info("  Saved scaler_%s.pkl", tag)

    train_ds = WindowDataset(train_runs, window_size, stride)
    val_ds   = WindowDataset(val_runs,   window_size, stride)
    test_ds  = WindowDataset(test_runs,  window_size, stride)

    log.info("  Windows: train=%s  val=%s  test=%s",
             f"{len(train_ds):,}", f"{len(val_ds):,}", f"{len(test_ds):,}")

    if len(train_ds) == 0 or len(test_ds) == 0:
        log.info("  Not enough windows — try smaller window/stride.")
        for d in (cache_train, cache_val, cache_test):
            shutil.rmtree(d, ignore_errors=True)
        return {}

    if len(val_ds) == 0:
        log.info("  Warning: no val windows. Reusing train windows for "
                 "validation to avoid test leakage.")
        val_ds = train_ds

    use_pin = device.type == "cuda"
    pw      = num_workers > 0
    # drop_last=True on the TRAIN loader only: nn.BatchNorm1d (used in the
    # MLP branch of every sequence architecture) raises ValueError if a
    # training-mode forward pass ever sees a batch of exactly 1 sample
    # ("Expected more than 1 value per channel when training") — which
    # happens whenever total_train_windows % batch_size == 1, entirely by
    # chance, and would otherwise crash training partway through an epoch
    # with no partial-result recovery. Guarded to only apply when there's
    # more than one full batch of data, so a small dataset doesn't end up
    # silently training on zero batches instead of raising or crashing.
    train_drop_last = len(train_ds) > batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw, drop_last=train_drop_last)
    # val/test are evaluated in model.eval() mode, where BatchNorm uses its
    # stored running statistics rather than the current batch's — so a
    # batch of size 1 is never a problem here. drop_last stays False (the
    # default) so every val/test window is always evaluated; dropping the
    # last partial batch here would silently shrink the reported test set.
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw)

    class_weights = compute_class_weights(train_runs, n_classes)

    all_details: dict = {}

    for mode in modes:
        log.info("")
        log.info("  — Model: %s —", mode.upper())

        # Reproducibility: seed immediately before building/training this
        # mode so its weight init and DataLoader shuffling are deterministic
        # given random_state, matching the guarantee the tabular family
        # already has via sklearn's own random_state. Re-seeding per mode
        # (not once for the whole function) also means training "lstm"
        # alone gives the same result whether or not "mlp" ran first in the
        # same call.
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        model = build_model(mode, n_features=n_features, n_classes=n_classes,
                            hidden_lstm=128, hidden_mlp=64,
                            n_lstm_layers=2, dropout=0.3)

        model, history = train_model(
            model, train_loader, val_loader, device,
            n_classes, class_weights,
            epochs=epochs, lr=1e-3, patience=patience,
        )

        plot_training_curves(
            history, f"{task_name} — {mode.upper()}",
            output_dir / f"curves_{tag}_{mode}.png",
        )

        log.info("  Evaluating on test set …")
        y_true, y_pred, y_prob, report, auc = evaluate_model(
            model, test_loader, device, class_names=class_names,
        )

        y_true_out, y_pred_out = _decode_labels(y_true, y_pred, le, is_cls)

        plot_confusion(
            y_true_out, y_pred_out, class_names,
            f"{task_name} — {mode.upper()}",
            output_dir / f"confusion_{tag}_{mode}.png",
        )

        if compute_shap:
            plot_attribution(
                model, test_loader, available,
                f"{task_name} — {mode.upper()} — Attribution",
                output_dir / f"shap_{tag}_{mode}.png",
                device, max_samples=shap_sample_size,
            )

        model_path = output_dir / f"model_{tag}_{mode}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "mode":             mode,
            "n_features":       n_features,
            "n_classes":        n_classes,
            "class_names":      class_names,
            "feature_names":    available,
            "window_size":      window_size,
            "stride":           stride,
            "hidden_lstm":      128,
            "hidden_mlp":       64,
            "n_lstm_layers":    2,
            "dropout":          0.3,
        }, model_path)
        log.info("  Saved: %s", model_path.name)

        all_details[mode] = {
            "model":         model,
            "report":        report,
            "auc":           auc,
            "label_encoder": le,
            "class_names":   class_names,
            "feature_names": available,
            "y_true":        y_true_out,
            "y_pred":        y_pred_out,
            "model_path":    str(model_path),
        }

    for cache_dir in (cache_train, cache_val, cache_test):
        shutil.rmtree(cache_dir, ignore_errors=True)

    return {m: (d if return_details else d["report"]) for m, d in all_details.items()}


# ---------------------------------------------------------------------------
#  Unified entry point — every model family, one call, one workflow
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    target_col: str,
    registry: list[dict],
    feature_cols: list[str],
    model_modes: list[str],
    *,
    output_dir,
    cfg,
    window_size: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    device=None,
    patience: int | None = None,
    num_workers: int = 0,
    train_entries: list[dict] | None = None,
    val_entries: list[dict] | None = None,
    test_entries: list[dict] | None = None,
    artifact_prefix: str | None = None,
    max_train_rows: int | None = 1_000_000,
    random_state: int = 42,
    tabular_kwargs: dict[str, dict] | None = None,
    save_deployment: bool = True,
    compute_shap: bool = False,
    shap_sample_size: int = 256,
    lookback_window: int = 12,
    custom_runners: dict[str, callable] | None = None,
    return_details: bool = False,
) -> dict:
    """Train every requested model family through one shared workflow.

    This is the single entry point for the entire training pipeline — the
    notebook calls this once per task (detection, classification) with
    whatever mix of tabular and sequence ``model_modes`` it wants compared,
    and gets back a uniform ``{mode: details_or_report}`` dict regardless
    of which family each mode came from.

    What is guaranteed identical across every mode in one call
    --------------------------------------------------------------
    * **Files** — ``train_entries`` / ``val_entries`` / ``test_entries``
      are resolved exactly once, from the same registry, before any model
      family is touched.
    * **Labels** — for classification, one ``LabelEncoder`` is fit here
      from real ``fault_type`` row values (via
      :func:`~library.data.loader.discover_fault_types`) across train+val
      entries, and the *same fitted encoder* is threaded into every
      backend. Previously the tabular and sequence backends each fit
      their own encoder independently — from different data (row values
      vs. filename metadata) and potentially different file sets (train
      vs. train+val) — which could silently disagree on which integer
      index a given fault type mapped to.
    * **Rows, within the tabular family** — every requested tabular
      estimator is fit against the identical, once-drawn stratified
      subsample (see :func:`_run_tabular_modes`).
    * **Windows, within the sequence family** — every requested
      architecture is fit against the identical memmap-backed windows
      (see :func:`_run_sequence_modes`); this sharing already existed
      before this revision.

    Sequence models cannot share row-level identity with the tabular
    family — they need contiguous per-file timesteps for windowing, which
    is incompatible with an arbitrary flat-row subsample — so "same
    files, same labels" is the cross-family guarantee; "same rows" is a
    within-family guarantee for tabular modes specifically.

    Parameters
    ----------
    task_name : str
        Human-readable name, e.g. ``"Fault Detection"``.
    target_col : str
        ``"fault_active"`` (binary detection) or ``"fault_type"``
        (multi-class classification).
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` +
        :func:`~library.data.assign_splits`, or loaded back via
        :func:`~library.data.load_registry_manifest`.
    feature_cols : list[str]
    model_modes : list[str]
        Any mix of :data:`~library.models.definitions.TABULAR_ESTIMATORS`
        keys (built-in: ``"random_forest"``) and
        :data:`~library.models.definitions.SEQUENCE_MODES`
        (``"mlp"``, ``"lstm"``, ``"hybrid"``), plus any keys present in
        ``custom_runners`` for model families that don't fit either shape.
    window_size, stride, batch_size, epochs, patience : int or None
        Required if any sequence mode is requested; ignored otherwise.
    device : torch.device or None
        Required if any sequence mode is requested.
    max_train_rows : int or None
        Cap on tabular training rows, applied once and shared by every
        tabular mode in this call (stratified by the target column where
        possible). ``None`` disables subsampling. Has no effect on
        sequence modes, which consume full per-file windows.
    random_state : int
        Seed for the shared tabular subsample and the default
        ``random_state`` of any tabular estimator that doesn't override it
        via ``tabular_kwargs``.
    tabular_kwargs : dict[str, dict] or None
        Per-mode estimator hyperparameter overrides, e.g.
        ``{"random_forest": {"n_estimators": 300, "max_depth": 20}}``.
        Deliberately separate from ``max_train_rows`` / ``random_state``:
        those two govern shared *data handling* and must be identical for
        every tabular mode in a call (that's what makes "no different
        sampling across models" a structural guarantee rather than a
        convention every caller has to remember); this dict is only for
        per-estimator *hyperparameters*.
    save_deployment : bool
        If True (default), also fit and save a SCADA+stress-only variant
        of every tabular model (works on real SCADA data with no
        SOLEY-simulation-only device physics).
    compute_shap : bool
        Opt-in gradient-attribution plot for sequence models — see
        :func:`plot_attribution`. Off by default; no effect on tabular
        modes (tabular attribution isn't implemented yet).
    shap_sample_size : int
        Number of test windows to run attribution over when
        ``compute_shap=True``.
    lookback_window : int
        Forwarded to every data read in this call (tabular and sequence
        alike) via :func:`~library.data.load_registry_entry`, keeping the
        rolling PR-deviation / power-step features identical across every
        model family.
    custom_runners : dict[str, callable] or None
        Extension point for model families that don't fit the tabular /
        sequence split. Each runner is called with the same keyword
        arguments (``task_name``, ``target_col``, ``registry``,
        ``feature_cols``, ``output_dir``, ``cfg``, ``train_entries``,
        ``val_entries``, ``test_entries``, ``artifact_prefix``,
        ``return_details``) for every mode key present in both
        ``model_modes`` and ``custom_runners``.
    return_details : bool
        If True, each mode's value in the returned dict is the full
        details dict (``model``, ``report``, ``auc``, ``label_encoder``,
        ``class_names``, ``feature_names``, ``y_true``, ``y_pred``,
        ``model_path``). If False, each mode's value is just the
        classification report dict.

    Returns
    -------
    dict
        ``{mode: details_or_report}`` — uniform across every model family
        that successfully trained. A mode is silently omitted (not set to
        ``None``) when its family had no entries to train on, or — for
        tabular modes specifically — when its estimator factory needed an
        optional dependency that wasn't installed.
    """
    from pathlib import Path

    from library.models.definitions import SEQUENCE_MODES, TABULAR_ESTIMATORS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    custom_runners = custom_runners or {}

    # --- 1. Resolve entries ONCE — every model family in this call trains
    # and evaluates on exactly the same files. ------------------------------
    if train_entries is None:
        train_entries = [e for e in registry if e.get("split") == "train"]
    if val_entries is None:
        val_entries = [e for e in registry if e.get("split") == "val"]
    if test_entries is None:
        test_entries = [e for e in registry if e.get("split") == "test"]

    is_cls = (target_col == "fault_type")
    if is_cls:
        train_entries = [e for e in train_entries if e["fault_type"] != "none"]
        val_entries   = [e for e in val_entries   if e["fault_type"] != "none"]
        test_entries  = [e for e in test_entries  if e["fault_type"] != "none"]

    if not train_entries or not test_entries:
        log.info("  No entries — skipping %s.", task_name)
        return {}

    sequence_modes = [m for m in model_modes if m in SEQUENCE_MODES]
    tabular_modes  = [m for m in model_modes if m in TABULAR_ESTIMATORS]
    custom_modes   = [m for m in model_modes
                      if m not in SEQUENCE_MODES | set(TABULAR_ESTIMATORS)]

    # --- 2. Fit ONE shared LabelEncoder from real data ----------------------
    le = None
    class_names = ["Healthy", "Faulted"]
    if is_cls:
        from sklearn.preprocessing import LabelEncoder

        from library.data.loader import discover_fault_types

        fault_types = discover_fault_types(train_entries + val_entries)
        if not fault_types:
            log.info("  No faulted rows in train/val — skipping %s.", task_name)
            return {}
        le = LabelEncoder()
        le.fit(fault_types)
        class_names = list(le.classes_)
        log.info("  Shared label encoder: %d classes — %s",
                 len(class_names), class_names)

    results: dict = {}

    # --- 3. Tabular family ---------------------------------------------------
    if tabular_modes:
        results.update(_run_tabular_modes(
            tabular_modes, task_name, target_col, is_cls,
            train_entries, val_entries, test_entries, feature_cols, cfg,
            le, class_names, output_dir, artifact_prefix, lookback_window,
            max_train_rows, random_state, tabular_kwargs, save_deployment,
            return_details,
        ))

    # --- 4. Sequence family ---------------------------------------------------
    if sequence_modes:
        missing = [
            name for name, value in {
                "window_size": window_size,
                "stride": stride,
                "batch_size": batch_size,
                "epochs": epochs,
                "device": device,
                "patience": patience,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Sequence modes require these parameters: " + ", ".join(missing)
            )

        results.update(_run_sequence_modes(
            sequence_modes, task_name, target_col, is_cls,
            train_entries, val_entries, test_entries, feature_cols, cfg,
            le, class_names, output_dir, artifact_prefix,
            window_size, stride, batch_size, epochs, device, patience,
            num_workers, lookback_window, compute_shap, shap_sample_size,
            random_state, return_details,
        ))

    # --- 5. Custom runners (extension point) ----------------------------------
    for mode in custom_modes:
        runner = custom_runners.get(mode)
        if runner is None:
            log.info("  Skipping unsupported model mode: %s", mode)
            continue

        custom_result = runner(
            task_name=task_name,
            target_col=target_col,
            registry=registry,
            feature_cols=feature_cols,
            output_dir=output_dir,
            cfg=cfg,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            artifact_prefix=artifact_prefix,
            return_details=return_details,
        )
        if isinstance(custom_result, dict) and mode in custom_result:
            results.update(custom_result)
        else:
            results[mode] = custom_result

    # --- 6. Cross-family per-fault F1 comparison (every requested mode) ------
    # Deliberately placed here, not inside _run_tabular_modes /
    # _run_sequence_modes: this is the one place with visibility into every
    # family's results at once. Previously this comparison lived only inside
    # the sequence-model path, so a call mixing tabular and sequence modes —
    # exactly what 02_unified_model_training.ipynb's MODEL_MODES does for
    # every study (main training, temporal split, cross-location, ablation)
    # — silently produced a plot/CSV that only ever showed mlp/lstm/hybrid,
    # never random_forest, even though random_forest's own report was
    # computed and returned correctly elsewhere. Building it here from the
    # combined `results` dict means it reflects everything actually trained.
    if is_cls and len(results) > 1:
        from library.visualization import plot_per_fault_f1

        reports_by_mode = {}
        for mode, value in results.items():
            report = value.get("report") if isinstance(value, dict) and "report" in value else value
            # Defensive: only include modes whose result actually looks like
            # a classification_report dict (class-name keys mapping to
            # {precision, recall, f1-score, ...} sub-dicts). A custom_runners
            # entry might not have this shape — skip it here rather than
            # crash the whole comparison over one differently-shaped mode.
            if isinstance(report, dict) and any(
                isinstance(v, dict) and "f1-score" in v for v in report.values()
            ):
                reports_by_mode[mode] = report

        if len(reports_by_mode) > 1:
            tag = artifact_prefix or task_name.lower().replace(" ", "_")
            plot_per_fault_f1(reports_by_mode, class_names,
                              output_dir / f"per_fault_f1_{tag}.png")

            rows = []
            for cls in class_names:
                row = {"fault_type": cls}
                for m, rep in reports_by_mode.items():
                    if cls in rep:
                        row[f"{m}_f1"]        = rep[cls].get("f1-score",  0)
                        row[f"{m}_precision"] = rep[cls].get("precision", 0)
                        row[f"{m}_recall"]    = rep[cls].get("recall",    0)
                rows.append(row)

            import pandas as pd
            # Tag included in the filename — unlike the pre-fix version,
            # which used a fixed "per_fault_metrics.csv" name that silently
            # overwrote itself across repeated run_task calls with different
            # artifact_prefix (e.g. once per fold in run_cross_location, or
            # once per feature set in run_feature_ablation).
            csv_path = output_dir / f"per_fault_metrics_{tag}.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            log.info("  Saved %s (%d model(s) compared)",
                     csv_path.name, len(reports_by_mode))

    return results


__all__ = [
    "run_task",
    "train_model",
    "evaluate_model",
    "compute_class_weights",
    "plot_attribution",
]
