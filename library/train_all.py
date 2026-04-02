"""
SOLEY ML Data Factory -- Complete Training Pipeline
====================================================

Single command to run the full ML training pipeline. Each model
trains in its own subprocess so that ALL memory is freed between
models. This prevents OOM kills on long runs with large datasets.

Pipeline:
  1. Random Forest (5 tasks: detection, classification, temporal,
     cross-location, feature ablation)
  2. PyTorch MLP      -- detection (separate process)
  3. PyTorch MLP      -- classification (separate process)
  4. PyTorch LSTM     -- detection (separate process)
  5. PyTorch LSTM     -- classification (separate process)
  6. PyTorch Hybrid   -- detection (separate process)
  7. PyTorch Hybrid   -- classification (separate process)

Each step is a fresh process. When it exits, the OS reclaims ALL
its memory. No fragmentation, no accumulation. Scales to any
dataset size that fits in RAM for a single model.

Output (all in one folder):
  Figures:  confusion_*.png, importance_*.png, curves_*.png,
            cross_location.png, feature_ablation.png, per_fault_f1_*.png
  Models:   rf_model_*.pkl, model_fault_*.pt, scaler_*.pkl
  Reports:  demo_report.txt, per_fault_metrics.csv

Usage:
    python ml_train_all.py                        # full pipeline
    python ml_train_all.py --quick                # fast test
    python ml_train_all.py --skip-rf              # PyTorch only
    python ml_train_all.py --skip-pytorch         # RF only
    python ml_train_all.py --skip-detection       # classification only
    python ml_train_all.py --skip-classification  # detection only
    python ml_train_all.py --output-dir results   # custom output

Requirements:
    pip install scikit-learn torch
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("soley_ml_all")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_step(cmd_args, step_name):
    """Run a training step as a subprocess.

    Returns True on success, False on failure.
    When the subprocess exits, ALL its memory is freed by the OS.
    """
    cmd = [sys.executable] + cmd_args
    log.info("")
    log.info("-" * 70)
    log.info("  %s", step_name)
    log.info("  %s", " ".join(cmd))
    log.info("-" * 70)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    minutes = elapsed / 60

    if result.returncode != 0:
        if result.returncode == -9:
            log.error("  OOM KILLED: %s (%.1f min)", step_name, minutes)
        else:
            log.error("  FAILED: %s (exit code %d, %.1f min)",
                      step_name, result.returncode, minutes)
        return False

    log.info("  DONE: %s (%.1f min)", step_name, minutes)
    return True


def kill_zombies():
    """Kill any lingering joblib/loky worker processes."""
    try:
        subprocess.run(["pkill", "-f", "joblib.externals.loky"],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def prompt_data_dir():
    """Prompt user for batch output directory if not provided via CLI."""
    while True:
        path = input("\nPath to SOLEY batch output folder: ").strip()
        if not path:
            continue
        p = Path(path)
        if not p.is_dir():
            print(f"  Directory not found: {p}")
            continue
        parquets = list(p.glob("*.parquet"))
        if not parquets:
            print(f"  No .parquet files found in {p}")
            continue
        config = p / "batch_config.json"
        if not config.exists():
            print(f"  Warning: no batch_config.json in {p} (will try anyway)")
        print(f"  Found {len(parquets)} parquet files in {p}")
        return str(p)


def main():
    parser = argparse.ArgumentParser(
        description="SOLEY ML -- Complete Training Pipeline",
    )
    parser.add_argument("--data-dir", default=None,
                        help="Path to batch output folder "
                             "(prompted interactively if omitted)")
    parser.add_argument("--output-dir", default="ml_results")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: fewer data, fewer epochs")
    parser.add_argument("--skip-rf", action="store_true",
                        help="Skip Random Forest")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip all PyTorch models")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip detection task (PyTorch only)")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip classification task (PyTorch only)")
    parser.add_argument("--max-rows-rf", type=int, default=2_000_000,
                        help="Max rows for RF (default: 2M)")
    parser.add_argument("--max-runs-pytorch", type=int, default=96,
                        help="Max runs for PyTorch (default: 96)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max epochs per PyTorch model (default: 30)")
    parser.add_argument("--pytorch-models", default="mlp,lstm,hybrid",
                        help="Comma-separated PyTorch models to train "
                             "(default: mlp,lstm,hybrid)")
    args = parser.parse_args()

    # Prompt for data dir if not provided
    if args.data_dir is None:
        args.data_dir = prompt_data_dir()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Resolve script paths relative to this file (works from any CWD)
    script_dir = Path(__file__).resolve().parent
    rf_script = str(script_dir / "ml_fault_demo.py")
    pt_script = str(script_dir / "ml_sequence_demo.py")

    t_start = time.time()
    results = []

    log.info("")
    log.info("=" * 70)
    log.info("  SOLEY ML Data Factory -- Complete Training Pipeline")
    log.info("  Data:   %s", Path(args.data_dir).resolve())
    log.info("  Output: %s", output_dir.resolve())
    log.info("  Strategy: each model runs in its own process (no OOM)")
    log.info("=" * 70)

    # ==================================================================
    #  STEP 1: Random Forest (all tasks in one process -- it's fast)
    # ==================================================================
    if not args.skip_rf:
        rf_args = [
            rf_script,
            "--data-dir", args.data_dir,
            "--output-dir", str(output_dir),
            "--max-rows", str(args.max_rows_rf),
        ]
        if args.quick:
            rf_args.append("--quick")

        ok = run_step(rf_args, "Random Forest (all 5 tasks)")
        results.append(("RF: all tasks", ok))
        kill_zombies()

    # ==================================================================
    #  STEP 2: PyTorch models -- ONE process per (model x task)
    # ==================================================================
    if not args.skip_pytorch:
        models = [m.strip() for m in args.pytorch_models.split(",")]
        tasks = []
        if not args.skip_detection:
            tasks.append("detection")
        if not args.skip_classification:
            tasks.append("classification")

        if not tasks:
            log.info("  No PyTorch tasks selected -- skipping.")
        else:
            for task in tasks:
                for model in models:
                    step_name = f"PyTorch {model.upper()} {task}"

                    pt_args = [
                        pt_script,
                        "--data-dir", args.data_dir,
                        "--output-dir", str(output_dir),
                        "--model", model,
                        "--task", task,
                        "--max-runs", str(args.max_runs_pytorch),
                        "--epochs", str(args.epochs),
                    ]
                    if args.quick:
                        pt_args.append("--quick")

                    ok = run_step(pt_args, step_name)
                    results.append((step_name, ok))

                    # Memory is freed when the subprocess exits.
                    # No kill_zombies needed -- PyTorch doesn't spawn workers.

    # ==================================================================
    #  SUMMARY
    # ==================================================================
    total_time = time.time() - t_start
    total_hours = total_time / 3600

    log.info("")
    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE  (%.1f hours)", total_hours)
    log.info("=" * 70)
    log.info("")

    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        log.info("  %-45s %s", name, status)

    log.info("")
    log.info("  %d passed, %d failed", n_pass, n_fail)
    log.info("")

    # List saved models
    models_on_disk = (sorted(output_dir.glob("*.pt"))
                      + sorted(output_dir.glob("*.pkl")))
    if models_on_disk:
        log.info("  Trained models:")
        for m in models_on_disk:
            size_mb = m.stat().st_size / (1024 * 1024)
            log.info("    %-45s  %.1f MB", m.name, size_mb)
        log.info("")

    # List figures
    figures = sorted(output_dir.glob("*.png"))
    if figures:
        log.info("  Figures (%d):", len(figures))
        for f in figures:
            log.info("    %s", f.name)
        log.info("")

    log.info("  All outputs: %s", output_dir.resolve())

    if n_fail > 0:
        log.info("")
        log.info("  To re-run only failed steps, use --skip flags.")
        log.info("  Example: python ml_train_all.py --skip-rf --skip-detection")
        sys.exit(1)


if __name__ == "__main__":
    main()
