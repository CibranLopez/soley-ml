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
    lookback_window: int = 12,
    max_train_rows: int | str | None = 1_000_000,
    max_test_rows: int | str | None = 1_000_000,
    majority_thin_factor: int = 1,
    custom_runners: dict[str, callable] | None = None,
    return_details: bool = False,
) -> dict:
    """Run one task across any supported model modes.

    Parameters mirror the existing PyTorch and Random Forest runners, but the
    notebook only needs one call site regardless of model family.

    Parameters
    ----------
    lookback_window : int
        Rolling-feature window forwarded unchanged to every backend so that
        :func:`~library.data.load_registry_entry` (and therefore
        :func:`~library.features.add_features`) uses the same value for
        every model family — RF, MLP, LSTM, and Hybrid.  Default ``12``.
    max_train_rows, max_test_rows : int, "auto", or None
        Row caps for the tabular family, forwarded to
        :func:`~library.models.trainer.run_task`. ``"auto"`` sizes the
        cap from currently available system memory instead of a fixed
        number — prefer this over a hand-picked int when you don't
        already know how much safely fits, and over ``None`` unless you
        specifically need every row regardless of memory. Both bound
        peak memory *during* the load rather than only after — previously
        this wrapper didn't expose either knob at all, so every call
        silently used ``run_task``'s defaults with no way for a notebook
        to lower them for a memory-constrained environment. Default
        ``1_000_000`` for each, matching ``run_task``'s own defaults.
    majority_thin_factor : int
        Forwarded to :func:`~library.models.trainer.run_task`. Thins
        whichever ``fault_active`` value is locally the majority within
        *training* files only (decided per file — not assumed to always
        be the healthy side; batches where faults dominate the row count
        are not unusual), every minority-class row always kept, and the
        test split never thinned — a redundancy cut on near-duplicate
        majority-class timesteps, not a class filter. Default ``1``: no
        thinning, identical behaviour to before this parameter existed.
        Has no effect on sequence modes (``mlp``/``lstm``/``hybrid``),
        which need contiguous per-file timesteps for windowing.
    """
    from .trainer import run_task

    # Backward-compatibility: preserve the old rf_params argument by mapping
    # it to the new per-mode tabular kwargs API.
    tabular_kwargs = None
    if rf_params:
        tabular_kwargs = {"random_forest": dict(rf_params)}

    return run_task(
        task_name=task_name,
        target_col=target_col,
        registry=registry,
        feature_cols=feature_cols,
        model_modes=model_modes,
        output_dir=output_dir,
        cfg=cfg,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        patience=patience,
        num_workers=num_workers,
        train_entries=train_entries,
        val_entries=val_entries,
        test_entries=test_entries,
        artifact_prefix=artifact_prefix,
        tabular_kwargs=tabular_kwargs,
        lookback_window=lookback_window,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
        majority_thin_factor=majority_thin_factor,
        custom_runners=custom_runners,
        return_details=return_details,
    )


__all__ = ["run_model_task", "SEQUENCE_MODEL_MODES", "TABULAR_MODEL_MODES"]