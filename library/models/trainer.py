"""
library.models.trainer
========================

PyTorch training loop and evaluation for fault detection / classification.

train_model(model, train_loader, val_loader, device, ...)
    Full training loop with:
      * Weighted cross-entropy (handles class imbalance)
      * Adam optimiser + ReduceLROnPlateau scheduler
      * Gradient clipping (max_norm=1)
      * Early stopping (patience-based)
    Returns the best model (lowest val loss) and a history dict.

evaluate_model(model, test_loader, device, class_names)
    Returns predictions, probabilities, a classification report dict,
    and ROC-AUC.

compute_class_weights(runs, n_classes)
    Scans memmap label files to compute inverse-frequency weights
    without loading all data into RAM.
"""

import logging

import numpy as np

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Class-weight computation (memmap-aware)
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
#  Training loop
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
        The model to train (created by :func:`~library.models.build_model`).
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
#  Evaluation
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
    from sklearn.metrics import classification_report, roc_auc_score

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

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    log.info("")
    log.info(report_str)

    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob,
                                multi_class="ovr", average="weighted")
        log.info("  ROC AUC: %.4f", auc)
    except Exception:
        auc = None

    return y_true, y_pred, y_prob, report, auc


# ---------------------------------------------------------------------------
#  Full task pipeline (scaler → memmap → DataLoader → train → evaluate)
# ---------------------------------------------------------------------------

def run_pytorch_task(
    task_name: str,
    target_col: str,
    registry: list[dict],
    feature_cols: list[str],
    model_modes: list[str],
    window_size: int,
    stride: int,
    batch_size: int,
    epochs: int,
    device,
    output_dir,
    patience: int,
    cfg,
    num_workers: int = 0,
) -> dict:
    """End-to-end PyTorch training pipeline for one task.

    Handles scaler fitting, memmap building, DataLoader creation,
    model training, evaluation, and artifact saving.

    Parameters
    ----------
    task_name : str
        Human-readable name, e.g. ``"Fault Detection"``.
    target_col : str
        ``"fault_active"`` or ``"fault_type"``.
    registry : list[dict]
        From :func:`~library.data.prepare_file_registry` + assign_splits.
    feature_cols : list[str]
    model_modes : list[str]
        Subset of ``["mlp", "lstm", "hybrid"]``.
    window_size, stride, batch_size, epochs, patience : int
    device : torch.device
    output_dir : Path
    cfg : BatchConfig
    num_workers : int

    Returns
    -------
    dict  — ``{mode: classification_report_dict}`` for each trained model.
    """
    import shutil

    import torch
    from torch.utils.data import DataLoader

    from library.models.dataset import (
        WindowDataset,
        build_memmap_arrays,
        fit_scaler_streaming,
    )
    from library.models.neural_network import build_model

    from library.visualization import (
        plot_confusion,
        plot_training_curves,
        plot_per_fault_f1,
    )

    import joblib
    from pathlib import Path

    output_dir = Path(output_dir)
    log.info("")
    log.info("=" * 65)
    log.info("  %s", task_name)
    log.info("=" * 65)

    is_cls = (target_col == "fault_type")
    tag    = task_name.lower().replace(" ", "_")

    # Filter registry
    if is_cls:
        entries = [e for e in registry
                   if e["fault_type"] != "none" and "split" in e]
    else:
        entries = [e for e in registry if "split" in e]

    if not entries:
        log.info("  No entries — skipping.")
        return {}

    # Label encoder
    le = None
    if is_cls:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(sorted({e["fault_type"] for e in entries}))
        class_names = list(le.classes_)
    else:
        class_names = ["Healthy", "Faulted"]
    n_classes = len(class_names)

    # Split
    train_e = [e for e in entries if e.get("split") == "train"]
    val_e   = [e for e in entries if e.get("split") == "val"]
    test_e  = [e for e in entries if e.get("split") == "test"]
    log.info("  Classes: %d  Files: train=%d, val=%d, test=%d",
             n_classes, len(train_e), len(val_e), len(test_e))

    # Scaler
    log.info("  Fitting scaler …")
    scaler, available = fit_scaler_streaming(train_e, feature_cols, cfg)
    n_features = len(available)

    # Memmap arrays
    log.info("  Building train arrays (%d files) …", len(train_e))
    train_runs, cache_train = build_memmap_arrays(
        train_e, available, target_col, cfg, scaler, le, is_cls)
    log.info("  Building val arrays (%d files) …", len(val_e))
    val_runs, cache_val = build_memmap_arrays(
        val_e, available, target_col, cfg, scaler, le, is_cls)
    log.info("  Building test arrays (%d files) …", len(test_e))
    test_runs, cache_test = build_memmap_arrays(
        test_e, available, target_col, cfg, scaler, le, is_cls)

    # Save preprocessing artifacts
    joblib.dump(scaler, output_dir / f"scaler_{tag}.pkl")
    if le is not None:
        joblib.dump(le, output_dir / f"label_encoder_{tag}.pkl")
    log.info("  Saved scaler_%s.pkl", tag)

    # Datasets and DataLoaders
    train_ds = WindowDataset(train_runs, window_size, stride)
    val_ds   = WindowDataset(val_runs,   window_size, stride)
    test_ds  = WindowDataset(test_runs,  window_size, stride)

    log.info("  Windows: train=%s  val=%s  test=%s",
             f"{len(train_ds):,}", f"{len(val_ds):,}", f"{len(test_ds):,}")

    if len(train_ds) == 0 or len(test_ds) == 0:
        log.info("  Not enough windows — try smaller window/stride.")
        for d in [cache_train, cache_val, cache_test]:
            shutil.rmtree(d, ignore_errors=True)
        return {}

    if len(val_ds) == 0:
        log.info("  Warning: no val windows. Using test for validation.")
        val_ds = test_ds

    use_pin      = device.type == "cuda"
    pw           = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=pw)

    # Class weights
    class_weights = compute_class_weights(train_runs, n_classes)

    all_reports = {}

    for mode in model_modes:
        log.info("")
        log.info("  — Model: %s —", mode.upper())

        model = build_model(n_features, n_classes, mode=mode,
                            hidden_lstm=128, hidden_mlp=64,
                            n_lstm_layers=2, dropout=0.3)

        model, history = train_model(
            model, train_loader, val_loader, device,
            n_classes, class_weights,
            epochs=epochs, lr=1e-3, patience=patience,
        )

        plot_training_curves(
            history,
            f"{task_name} — {mode.upper()}",
            output_dir / f"curves_{tag}_{mode}.png",
        )

        log.info("  Evaluating on test set …")
        y_true, y_pred, y_prob, report, auc = evaluate_model(
            model, test_loader, device, class_names=class_names,
        )

        plot_confusion(
            y_true, y_pred, class_names,
            f"{task_name} — {mode.upper()}",
            output_dir / f"confusion_{tag}_{mode}.png",
        )

        # Save model checkpoint
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

        all_reports[mode] = report

    # Per-fault F1 comparison across modes
    if len(all_reports) > 1 and is_cls:
        plot_per_fault_f1(
            all_reports, class_names,
            output_dir / f"per_fault_f1_{tag}.png",
        )
        import pandas as pd
        rows = []
        for cls in class_names:
            row = {"fault_type": cls}
            for m, rep in all_reports.items():
                if cls in rep:
                    row[f"{m}_f1"]        = rep[cls].get("f1-score",  0)
                    row[f"{m}_precision"] = rep[cls].get("precision", 0)
                    row[f"{m}_recall"]    = rep[cls].get("recall",    0)
            rows.append(row)
        csv_path = output_dir / "per_fault_metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info("  Saved %s", csv_path.name)

    # Cleanup temp dirs
    for cache_dir in [cache_train, cache_val, cache_test]:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return all_reports
