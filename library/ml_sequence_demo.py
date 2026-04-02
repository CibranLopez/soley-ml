"""
SOLEY ML Data Factory -- Sequence Model Demo (PyTorch)
======================================================

Companion to ml_fault_demo.py (Random Forest baseline).

Trains LSTM-based models on temporal WINDOWS of SOLEY batch data.
The model sees consecutive timesteps and learns temporal fault
signatures -- gradual degradation, recurring events, progressive
parameter shifts.

Three model variants for paper comparison:
  - mlp:    Per-timestep features only (neural network baseline)
  - lstm:   Temporal window only (sequence model)
  - hybrid: Both branches combined (best of both worlds)

Auto-configures from batch_config.json and parquet schema.
Works with ANY SOLEY batch output -- no hardcoded values.

Memory-safe: files are streamed one at a time to disk-backed memmap
arrays. Training windows are paged from memmap on demand. Peak RAM
equals one parquet file (~60-100 MB), regardless of dataset size.
All 774+ runs train without hitting memory limits.

Usage:
    python ml_sequence_demo.py                           # hybrid, all data
    python ml_sequence_demo.py --model all               # all three
    python ml_sequence_demo.py --model all --task both   # full pipeline
    python ml_sequence_demo.py --quick                   # 1 location
    python ml_sequence_demo.py --max-runs 48             # limit data

Requirements:
    pip install torch scikit-learn
"""

import argparse
import bisect
import gc
import logging
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_utils import (
    BatchConfig, add_features, plot_confusion,
)

log = logging.getLogger("soley_seq")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ===================================================================
#  FILE REGISTRY (metadata from filenames — no data loaded)
# ===================================================================

def _extract_fault_type(filepath):
    """Extract fault type from parquet filename without loading data."""
    name = Path(filepath).stem
    for suffix in ("_noisy", "_clean"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    if "_fault_" in name:
        after = name.split("_fault_")[1]
        for sep in ("_af", "_aging"):
            if sep in after:
                return after.split(sep)[0]
        return after
    elif "_combo" in name:
        return "combo"
    return "none"


def _prepare_file_registry(data_dir, cfg, max_runs=None,
                            location_filter=None):
    """Discover parquet files and extract metadata. No data loaded."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*_noisy.parquet"))
    if not files:
        files = sorted(data_dir.glob("*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    if location_filter:
        files = [f for f in files if location_filter in f.name]

    registry = []
    for f in files:
        run_id = f.stem.split("_")[0]
        ft = _extract_fault_type(f)
        registry.append({"path": f, "run_id": run_id, "fault_type": ft})

    # Balance across fault types if limited
    if max_runs and len(registry) > max_runs:
        groups = defaultdict(list)
        for e in registry:
            groups[e["fault_type"]].append(e)
        per_type = max(1, max_runs // len(groups))
        selected = []
        for ft in sorted(groups):
            selected.extend(groups[ft][:per_type])
        registry = sorted(selected,
                          key=lambda e: str(e["path"]))[:max_runs]

    fault_types = sorted(set(e["fault_type"] for e in registry))
    log.info("Registry: %d files, %d fault types",
             len(registry), len(fault_types))
    return registry


def _assign_splits(registry, seed=42):
    """Stratified train/val/test split by fault type. No data loaded."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for e in registry:
        groups[e["fault_type"]].append(e)

    for ft, entries in groups.items():
        rng.shuffle(entries)
        n = len(entries)
        if n >= 3:
            n_train = max(1, int(0.6 * n))
            n_val = max(1, int(0.2 * n))
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
            for e in entries[:n_train]:
                e["split"] = "train"
            for e in entries[n_train:n_train + n_val]:
                e["split"] = "val"
            for e in entries[n_train + n_val:]:
                e["split"] = "test"
        elif n == 2:
            entries[0]["split"] = "train"
            entries[1]["split"] = "test"
        else:
            entries[0]["split"] = "train"

    for s in ("train", "val", "test"):
        count = sum(1 for e in registry if e.get("split") == s)
        log.info("  %s: %d files", s, count)


# ===================================================================
#  STREAMING SCALER + MEMMAP BUILDER
# ===================================================================

def _fit_scaler_streaming(entries, feature_cols, cfg, n_sample_files=20):
    """Fit StandardScaler from sampled rows across a few files.

    Peak RAM = 1 file at a time.
    """
    from sklearn.preprocessing import StandardScaler

    step = max(1, len(entries) // n_sample_files)
    sample_entries = entries[::step][:n_sample_files]

    chunks = []
    available = None

    for e in sample_entries:
        df = pd.read_parquet(e["path"], columns=cfg.load_cols)
        df = add_features(df, cfg.array_kwp)
        if available is None:
            available = [c for c in feature_cols if c in df.columns]
        n = min(10_000, len(df))
        if n > 0:
            chunks.append(df[available].sample(n=n, random_state=42))
        del df

    combined = pd.concat(chunks).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaler.fit(combined.values.astype(np.float32))
    del chunks, combined
    gc.collect()

    log.info("  Scaler fit on %d files, %d features",
             len(sample_entries), len(available))
    return scaler, available


def _build_runs_streaming(entries, feature_cols, target_col, cfg,
                           scaler, label_encoder=None, faulted_only=False):
    """Stream parquet files to memmap one at a time.

    Peak RAM = 1 parquet file (~60-100 MB). All converted arrays
    live on disk as memory-mapped files, paged in on demand.

    To avoid hitting OS file descriptor limits, arrays are written
    to disk during conversion but NOT kept open. The returned list
    contains (path, shape, label_path, label_shape, run_id) tuples.
    Memmap handles are opened lazily by WindowDataset.__getitem__.
    """
    cache_dir = Path(tempfile.mkdtemp(prefix="soley_ml_"))
    runs = []  # (feat_path, feat_shape, lab_path, lab_shape, run_id)

    for i, e in enumerate(entries):
        df = pd.read_parquet(e["path"], columns=cfg.load_cols)
        df = add_features(df, cfg.array_kwp)

        if faulted_only and "fault_active" in df.columns:
            df = df[df["fault_active"]].copy()
        if len(df) == 0:
            del df
            continue

        df = df.sort_values("timestamp")

        # Features -> scaled float32
        feat = df[feature_cols].values.astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = scaler.transform(feat).astype(np.float32)

        # Labels
        if target_col == "fault_active":
            labels = df["fault_active"].astype(int).values.astype(np.int64)
        else:
            labels = label_encoder.transform(
                df["fault_type"].values
            ).astype(np.int64)

        # Write to disk, then close immediately (no open fd kept)
        fp = cache_dir / f"feat_{i}.dat"
        lp = cache_dir / f"lab_{i}.dat"

        fm = np.memmap(fp, dtype=np.float32, mode="w+", shape=feat.shape)
        fm[:] = feat
        fm.flush()
        feat_shape = feat.shape
        del fm

        lm = np.memmap(lp, dtype=np.int64, mode="w+", shape=labels.shape)
        lm[:] = labels
        lm.flush()
        lab_shape = labels.shape
        del lm

        # Store paths + shapes, NOT open memmap handles
        runs.append((str(fp), feat_shape, str(lp), lab_shape, e["run_id"]))

        del df, feat, labels
        if (i + 1) % 50 == 0:
            gc.collect()
            log.info("    ... %d / %d files", i + 1, len(entries))

    gc.collect()
    log.info("  Cached %d runs to disk", len(runs))
    return runs, cache_dir


# ===================================================================
#  WINDOWING (compact index — ~800 entries instead of ~15M)
# ===================================================================

class WindowDataset:
    """PyTorch dataset: temporal windows from disk-backed arrays.

    Uses a compact index with one entry per run instead of one tuple
    per window. For 774 runs with stride=12 this saves ~1.5 GB of
    Python object overhead.

    Memmap files are opened lazily per __getitem__ call and cached
    with an LRU policy to avoid hitting OS file descriptor limits.
    """

    def __init__(self, run_arrays, window_size, stride):
        """
        Parameters
        ----------
        run_arrays : list of (feat_path, feat_shape, lab_path, lab_shape, run_id)
        """
        self.run_arrays = run_arrays
        self.window_size = window_size
        self.stride = stride

        self._run_info = []      # (run_idx, n_windows)
        self._cumulative = []    # cumulative window count before this run
        self._total = 0

        for run_idx, (fp, fs, lp, ls, rid) in enumerate(run_arrays):
            n = fs[0]  # number of timesteps
            if n < window_size:
                continue
            n_windows = (n - window_size) // stride + 1
            self._run_info.append((run_idx, n_windows))
            self._cumulative.append(self._total)
            self._total += n_windows

        # LRU cache for open memmap handles (limit open fds)
        self._cache = {}
        self._cache_order = []
        self._cache_max = 64

    def _get_arrays(self, run_idx):
        """Get (feat_memmap, lab_memmap) for a run, with LRU caching."""
        if run_idx in self._cache:
            return self._cache[run_idx]

        fp, fs, lp, ls, _ = self.run_arrays[run_idx]
        feat = np.memmap(fp, dtype=np.float32, mode="r", shape=fs)
        lab = np.memmap(lp, dtype=np.int64, mode="r", shape=ls)

        # Evict oldest if cache full
        if len(self._cache_order) >= self._cache_max:
            evict_idx = self._cache_order.pop(0)
            old = self._cache.pop(evict_idx, None)
            if old is not None:
                del old  # closes memmap fd

        self._cache[run_idx] = (feat, lab)
        self._cache_order.append(run_idx)
        return feat, lab

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        import torch
        # Binary search: which run does this global index fall in?
        pos = bisect.bisect_right(self._cumulative, idx) - 1
        run_idx, _ = self._run_info[pos]
        local_idx = idx - self._cumulative[pos]
        start = local_idx * self.stride

        feat, labels = self._get_arrays(run_idx)
        end = start + self.window_size
        window = torch.from_numpy(feat[start:end].copy())
        label = torch.tensor(int(labels[end - 1]), dtype=torch.long)
        return window, label


# ===================================================================
#  MODEL
# ===================================================================

def build_model(n_features, n_classes, mode="hybrid",
                hidden_lstm=128, hidden_mlp=64, n_lstm_layers=2,
                dropout=0.3):
    """Build MLP / LSTM / Hybrid fault model."""
    import torch
    import torch.nn as nn

    class HybridFaultModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode

            if mode in ("mlp", "hybrid"):
                self.mlp = nn.Sequential(
                    nn.Linear(n_features, hidden_mlp),
                    nn.BatchNorm1d(hidden_mlp),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_mlp, hidden_mlp),
                    nn.ReLU(),
                )

            if mode in ("lstm", "hybrid"):
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_lstm,
                    num_layers=n_lstm_layers,
                    batch_first=True,
                    dropout=dropout if n_lstm_layers > 1 else 0,
                )
                self.lstm_norm = nn.LayerNorm(hidden_lstm)

            head_in = 0
            if mode in ("mlp", "hybrid"):
                head_in += hidden_mlp
            if mode in ("lstm", "hybrid"):
                head_in += hidden_lstm

            self.head = nn.Sequential(
                nn.Linear(head_in, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x_window):
            parts = []
            if self.mode in ("mlp", "hybrid"):
                x_last = x_window[:, -1, :]
                parts.append(self.mlp(x_last))
            if self.mode in ("lstm", "hybrid"):
                lstm_out, _ = self.lstm(x_window)
                lstm_last = lstm_out[:, -1, :]
                parts.append(self.lstm_norm(lstm_last))
            combined = torch.cat(parts, dim=1)
            return self.head(combined)

    return HybridFaultModel()


# ===================================================================
#  TRAINING
# ===================================================================

def train_model(model, train_loader, val_loader, device,
                n_classes, class_weights, epochs=30, lr=1e-3,
                patience=7):
    """PyTorch training loop with early stopping."""
    import torch
    import torch.nn as nn

    model = model.to(device)

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32,
                                     device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        n_val_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        log.info("  Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  "
                 "val_acc=%.4f  lr=%.1e",
                 epoch, epochs, avg_train_loss, avg_val_loss,
                 val_acc, current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("  Early stopping at epoch %d (patience=%d)",
                         epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ===================================================================
#  EVALUATION
# ===================================================================

def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate and return predictions + metrics."""
    import torch
    from sklearn.metrics import classification_report, roc_auc_score

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
            all_probs.append(probs)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    target_names = class_names if class_names else None
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0,
    )
    log.info("")
    log.info(report_str)

    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="weighted")
        log.info("  ROC AUC: %.4f", auc)
    except Exception:
        auc = None

    return y_true, y_pred, y_prob, report, auc


# ===================================================================
#  PLOTTING
# ===================================================================

def plot_training_curves(history, title, path):
    """Plot loss and accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train", color="#2563eb")
    ax1.plot(epochs, history["val_loss"], label="Validation", color="#dc2626")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="Validation", color="#10b981")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)


def plot_per_fault_f1(results_by_model, class_names, path):
    """Per-fault-type F1 comparison across model variants."""
    models = list(results_by_model.keys())
    n_faults = len(class_names)
    x = np.arange(n_faults)
    width = 0.8 / len(models)
    colors = {"mlp": "#2563eb", "lstm": "#f59e0b", "hybrid": "#10b981"}

    fig, ax = plt.subplots(figsize=(max(10, n_faults * 1.2), 5))

    for i, model_name in enumerate(models):
        report = results_by_model[model_name]
        f1_scores = [report.get(cls, {}).get("f1-score", 0.0)
                     for cls in class_names]
        ax.bar(x + i * width, f1_scores, width,
               label=model_name.upper(),
               color=colors.get(model_name, "#666666"))

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
    log.info("  Saved %s", path.name)


# ===================================================================
#  MAIN PIPELINE
# ===================================================================

def run_task(task_name, target_col, registry, feature_cols, model_modes,
             window_size, stride, batch_size, epochs, device,
             output_dir, patience, cfg, num_workers=4):
    """Train and evaluate models with streaming data pipeline.

    Files are loaded one at a time, feature-engineered, scaled, and
    written to disk-backed memmap arrays. Peak RAM = 1 file (~60 MB).
    """
    import torch
    from torch.utils.data import DataLoader

    log.info("")
    log.info("=" * 65)
    log.info("  %s", task_name)
    log.info("=" * 65)

    is_cls = (target_col == "fault_type")

    # Filter registry for classification (exclude normal runs)
    if is_cls:
        entries = [e for e in registry
                   if e["fault_type"] != "none" and "split" in e]
    else:
        entries = [e for e in registry if "split" in e]

    if not entries:
        log.info("  No entries -- skipping.")
        return {}

    # Label encoder for classification
    le = None
    if is_cls:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(sorted(set(e["fault_type"] for e in entries)))
        class_names = list(le.classes_)
    else:
        class_names = ["Healthy", "Faulted"]

    n_classes = len(class_names)

    # Split entries
    train_e = [e for e in entries if e["split"] == "train"]
    val_e = [e for e in entries if e["split"] == "val"]
    test_e = [e for e in entries if e["split"] == "test"]

    log.info("  Classes: %d  Files: train=%d, val=%d, test=%d",
             n_classes, len(train_e), len(val_e), len(test_e))

    # Fit scaler on train split only
    log.info("  Fitting scaler (streaming) ...")
    scaler, available = _fit_scaler_streaming(train_e, feature_cols, cfg)
    n_features = len(available)

    # Stream files to memmap (1 file at a time)
    log.info("  Building train arrays (%d files) ...", len(train_e))
    train_runs, cache_train = _build_runs_streaming(
        train_e, available, target_col, cfg, scaler, le, is_cls)
    log.info("  Building val arrays (%d files) ...", len(val_e))
    val_runs, cache_val = _build_runs_streaming(
        val_e, available, target_col, cfg, scaler, le, is_cls)
    log.info("  Building test arrays (%d files) ...", len(test_e))
    test_runs, cache_test = _build_runs_streaming(
        test_e, available, target_col, cfg, scaler, le, is_cls)

    # Save scaler + label encoder
    import joblib
    tag = task_name.lower().replace(" ", "_")
    joblib.dump(scaler, output_dir / f"scaler_{tag}.pkl")
    if le is not None:
        joblib.dump(le, output_dir / f"label_encoder_{tag}.pkl")
    log.info("  Saved scaler: scaler_%s.pkl", tag)

    # Build windowed datasets (compact index)
    train_ds = WindowDataset(train_runs, window_size, stride)
    val_ds = WindowDataset(val_runs, window_size, stride)
    test_ds = WindowDataset(test_runs, window_size, stride)

    log.info("  Features: %d  Windows: train=%s  val=%s  test=%s",
             n_features,
             f"{len(train_ds):,}", f"{len(val_ds):,}", f"{len(test_ds):,}")

    if len(train_ds) == 0 or len(test_ds) == 0:
        log.info("  Not enough windows -- try smaller window/stride.")
        for d in [cache_train, cache_val, cache_test]:
            shutil.rmtree(d, ignore_errors=True)
        return {}

    if len(val_ds) == 0:
        log.info("  Warning: no val windows. Using test for validation.")
        val_ds = test_ds

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=use_pin, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=use_pin, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=use_pin, persistent_workers=num_workers > 0)

    # Class weights from memmap labels (opens one file at a time)
    class_counts = np.zeros(n_classes, dtype=np.float64)
    total_labels = 0
    for fp, fs, lp, ls, _ in train_runs:
        lab = np.memmap(lp, dtype=np.int64, mode="r", shape=ls)
        counts = np.bincount(np.asarray(lab), minlength=n_classes)
        class_counts += counts.astype(np.float64)
        total_labels += len(lab)
        del lab
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = total_labels / (n_classes * class_counts)

    # Train each model variant
    all_reports = {}

    for mode in model_modes:
        log.info("")
        log.info("-" * 50)
        log.info("  Model: %s", mode.upper())
        log.info("-" * 50)

        model = build_model(
            n_features, n_classes, mode=mode,
            hidden_lstm=128, hidden_mlp=64,
            n_lstm_layers=2, dropout=0.3,
        )

        n_params = sum(p.numel() for p in model.parameters())
        log.info("  Parameters: %s", f"{n_params:,}")

        model, history = train_model(
            model, train_loader, val_loader, device,
            n_classes, class_weights, epochs=epochs,
            lr=1e-3, patience=patience,
        )

        plot_training_curves(
            history, f"{task_name} -- {mode.upper()}",
            output_dir / f"curves_{tag}_{mode}.png",
        )

        log.info("  Evaluating on test set ...")
        y_true, y_pred, y_prob, report, auc = evaluate_model(
            model, test_loader, device, class_names=class_names,
        )

        plot_confusion(
            y_true, y_pred, class_names,
            f"{task_name} -- {mode.upper()}",
            output_dir / f"confusion_{tag}_{mode}.png",
        )

        # Save trained model
        model_path = output_dir / f"model_{tag}_{mode}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_names": class_names,
            "feature_names": available,
            "window_size": window_size,
            "hidden_lstm": 128,
            "hidden_mlp": 64,
            "n_lstm_layers": 2,
            "dropout": 0.3,
        }, model_path)
        log.info("  Saved model: %s", model_path.name)

        all_reports[mode] = report

    # Per-fault-type comparison
    if len(all_reports) > 1 and is_cls:
        plot_per_fault_f1(
            all_reports, class_names,
            output_dir / f"per_fault_f1_{tag}.png",
        )

        rows = []
        for cls in class_names:
            row = {"fault_type": cls}
            for mode, report in all_reports.items():
                if cls in report:
                    row[f"{mode}_f1"] = report[cls].get("f1-score", 0)
                    row[f"{mode}_precision"] = report[cls].get("precision", 0)
                    row[f"{mode}_recall"] = report[cls].get("recall", 0)
            rows.append(row)
        csv_path = output_dir / "per_fault_metrics.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info("  Saved %s", csv_path.name)

    # Cleanup temp cache dirs
    for cache_dir in [cache_train, cache_val, cache_test]:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return all_reports


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SOLEY ML -- Sequence Model Demo (PyTorch)",
    )
    parser.add_argument("--data-dir", default="batch_output")
    parser.add_argument("--output-dir", default="ml_sequence_results")
    parser.add_argument("--model", default="hybrid",
                        choices=["mlp", "lstm", "hybrid", "all"])
    parser.add_argument("--window", type=int, default=288,
                        help="Window size in timesteps (default: 288 = 24 hours)")
    parser.add_argument("--stride", type=int, default=12,
                        help="Stride between windows (default: 12 = 1 hour)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Limit files loaded (0=all, default: all)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for parallel data loading "
                             "(default: 0 = auto-detect CPU cores)")
    parser.add_argument("--task", default="both",
                        choices=["detection", "classification", "both"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Auto-detect configuration
    log.info("Auto-detecting batch configuration ...")
    cfg = BatchConfig(args.data_dir)

    # Device + CPU parallelism
    import os
    import torch
    n_cpus = os.cpu_count() or 4
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        # Use all cores for PyTorch matrix ops (BLAS/OpenMP)
        torch.set_num_threads(n_cpus)
        log.info("Using CPU: %d cores (torch threads=%d)",
                 n_cpus, torch.get_num_threads())

    # DataLoader workers: auto-detect from CPU count if not specified
    num_workers = args.num_workers
    if num_workers == 0:
        # Use half the cores for data loading (rest for torch compute)
        num_workers = max(2, n_cpus // 2)
    log.info("DataLoader workers: %d", num_workers)

    # Model modes
    model_modes = (["mlp", "lstm", "hybrid"] if args.model == "all"
                   else [args.model])

    # Feature list from auto-detected columns + engineered features
    engineered = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                  "performance_ratio", "pr_deviation",
                  "dc_ac_power_ratio", "power_step"]
    feature_list = (list(cfg.scada_features) + engineered
                    + list(cfg.device_features) + list(cfg.stress_features))

    # Location filter
    loc_filter = None
    if args.quick and cfg.locations:
        first_loc = list(cfg.locations.keys())[0]
        loc_filter = f"{first_loc[0]}_{first_loc[1]}"

    # Build file registry (NO data loaded — just file paths + metadata)
    max_runs = args.max_runs if args.max_runs > 0 else None
    registry = _prepare_file_registry(
        args.data_dir, cfg, max_runs=max_runs,
        location_filter=loc_filter)

    # Assign train/val/test splits from filenames
    _assign_splits(registry)

    # Tasks (data streamed to memmap inside run_task)
    if args.task in ("detection", "both"):
        run_task(
            "Fault Detection", "fault_active", registry, feature_list,
            model_modes, args.window, args.stride, args.batch_size,
            args.epochs, device, output_dir, args.patience, cfg,
            num_workers,
        )

    if args.task in ("classification", "both"):
        run_task(
            "Fault Classification", "fault_type", registry, feature_list,
            model_modes, args.window, args.stride, args.batch_size,
            args.epochs, device, output_dir, args.patience, cfg,
            num_workers,
        )

    log.info("")
    log.info("=" * 65)
    log.info("  COMPLETE")
    log.info("=" * 65)
    log.info("  Window: %d timesteps (%d min)",
             args.window, args.window * 5)
    log.info("  Models: %s", ", ".join(m.upper() for m in model_modes))
    log.info("  Device: %s", device)
    log.info("  Results: %s", output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        log.error("Install with:  pip install torch scikit-learn")
        sys.exit(1)
