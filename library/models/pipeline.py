"""
library.models.pipeline
=======================

Unified model runner for tabular and sequence architectures.

The notebook-facing API is a single function, :func:`run_model_task`, which
dispatches each requested ``model_mode`` to the appropriate backend while
keeping the same registry, split policy, output structure, and return format.
"""

import logging

log = logging.getLogger("library")

SEQUENCE_MODEL_MODES = {"mlp", "lstm", "hybrid"}
TABULAR_MODEL_MODES = {"random_forest"}


def run_model_task(
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
    rf_params: dict | None = None,
    custom_runners: dict[str, callable] | None = None,
    return_details: bool = False,
) -> dict:
    """Run one task across any supported model modes.

    Parameters mirror the existing PyTorch and Random Forest runners, but the
    notebook only needs one call site regardless of model family.
    """
    from .random_forest import run_rf_task
    from .trainer import run_pytorch_task

    rf_params = rf_params or {}
    custom_runners = custom_runners or {}

    sequence_modes = [m for m in model_modes if m in SEQUENCE_MODEL_MODES]
    tabular_modes = [m for m in model_modes if m in TABULAR_MODEL_MODES]
    custom_modes = [m for m in model_modes
                    if m not in SEQUENCE_MODEL_MODES | TABULAR_MODEL_MODES]

    results = {}

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

        results.update(run_pytorch_task(
            task_name=task_name,
            target_col=target_col,
            registry=registry,
            feature_cols=feature_cols,
            model_modes=sequence_modes,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            output_dir=output_dir,
            patience=patience,
            cfg=cfg,
            num_workers=num_workers,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            artifact_prefix=artifact_prefix,
            return_details=return_details,
        ))

    if tabular_modes:
        results.update(run_rf_task(
            task_name=task_name,
            target_col=target_col,
            registry=registry,
            feature_cols=feature_cols,
            cfg=cfg,
            output_dir=output_dir,
            train_entries=train_entries,
            val_entries=val_entries,
            test_entries=test_entries,
            artifact_prefix=artifact_prefix,
            return_details=return_details,
            **rf_params,
        ))

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

    return results


__all__ = ["run_model_task", "SEQUENCE_MODEL_MODES", "TABULAR_MODEL_MODES"]