"""
library.models.definitions
============================

Single model factory for every architecture this library supports —
tabular (scikit-learn-API estimators) and sequence (PyTorch nn.Module).

This replaces the old ``library.models.random_forest`` (which hardcoded
``RandomForestClassifier`` directly into the training function — a file
that could never logically contain anything *but* a random forest) and
``library.models.neural_network`` (same problem, one level up: a file
that could only ever build the three sequence architectures). Naming a
file after the one algorithm it happened to contain made the architecture
itself resistant to having a second tabular estimator, or a fourth
sequence architecture, added later. There is now exactly one place that
knows how to construct *any* supported model — ``build_model(mode, ...)``
— and exactly one dict that decides what counts as a "tabular" mode
(``TABULAR_ESTIMATORS``) versus a "sequence" mode (``SEQUENCE_MODES``);
:mod:`library.models.trainer` dispatches off these same two collections,
so adding a model here automatically makes it trainable, evaluable, and
comparable through the existing pipeline with no other file touched.

TABULAR_ESTIMATORS : dict[str, Callable]
    mode name -> zero/kwarg factory returning a fitted-ready, scikit-learn
    API estimator (``.fit`` / ``.predict`` / ``.predict_proba``). Currently
    one entry (``"random_forest"``); adding another tabular model is one
    function + one dict entry, not a new file.

SEQUENCE_MODES : set[str]
    ``{"mlp", "lstm", "hybrid"}`` — the PyTorch architectures, all built
    by the single ``HybridFaultModel`` class gated on ``mode``.

build_model(mode, *, n_features=None, n_classes=None, random_state=42, **kw)
    The unified entry point. Dispatches to either family based on which
    collection ``mode`` belongs to — callers don't need to know or care
    which family a given mode name is from.
"""

import logging

log = logging.getLogger("library")


# ---------------------------------------------------------------------------
#  Tabular estimators (scikit-learn API)
# ---------------------------------------------------------------------------

def _build_random_forest(*, n_estimators=200, max_depth=15,
                         min_samples_leaf=20, random_state=42, **extra):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        **extra,
    )


# mode name -> factory.  Adding a new tabular model is one function above
# plus one line here — nothing else in the library needs to change for it
# to become trainable via run_task(model_modes=[..., "your_new_mode"]).
TABULAR_ESTIMATORS = {
    "random_forest": _build_random_forest,
}


# ---------------------------------------------------------------------------
#  Sequence architectures (PyTorch nn.Module)
# ---------------------------------------------------------------------------

SEQUENCE_MODES = {"mlp", "lstm", "hybrid"}


def build_sequence_model(
    n_features: int,
    n_classes: int,
    mode: str = "hybrid",
    hidden_lstm: int = 128,
    hidden_mlp: int = 64,
    n_lstm_layers: int = 2,
    dropout: float = 0.3,
):
    """Construct a fault detection / classification sequence model.

    mode="mlp"     Per-timestep MLP on the last timestep of the window.
    mode="lstm"    Bidirectional LSTM processing the full window.
    mode="hybrid"  MLP branch (last step) + LSTM branch (all steps) fused.

    All three variants share the same ``forward(x_window)`` signature.

    State-of-the-art touches
    -------------------------
    * Layer-norm after LSTM (stabilises training with variable-length inputs).
    * Gradient clipping (max_norm=1) in the training loop.
    * ReduceLROnPlateau scheduler.
    * Weighted cross-entropy for class-imbalanced fault data.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_classes : int
        Number of output classes (2 for detection, N for classification).
    mode : {"mlp", "lstm", "hybrid"}
    hidden_lstm : int
        LSTM hidden size.
    hidden_mlp : int
        MLP hidden size (per layer).
    n_lstm_layers : int
    dropout : float

    Returns
    -------
    torch.nn.Module
    """
    import torch
    import torch.nn as nn

    class HybridFaultModel(nn.Module):
        """MLP / LSTM / Hybrid model for PV fault analysis."""

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
                    dropout=dropout if n_lstm_layers > 1 else 0.0,
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
            """
            Parameters
            ----------
            x_window : Tensor  shape (batch, window, n_features)

            Returns
            -------
            logits : Tensor  shape (batch, n_classes)
            """
            parts = []
            if self.mode in ("mlp", "hybrid"):
                x_last = x_window[:, -1, :]         # last timestep
                parts.append(self.mlp(x_last))
            if self.mode in ("lstm", "hybrid"):
                lstm_out, _ = self.lstm(x_window)
                lstm_last   = lstm_out[:, -1, :]    # last hidden state
                parts.append(self.lstm_norm(lstm_last))
            combined = torch.cat(parts, dim=1)
            return self.head(combined)

    model = HybridFaultModel()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Built model: mode=%s  params=%s", mode, f"{n_params:,}")
    return model


# ---------------------------------------------------------------------------
#  Single entry point — dispatches to whichever family `mode` belongs to
# ---------------------------------------------------------------------------

def build_model(
    mode: str,
    *,
    n_features: int | None = None,
    n_classes: int | None = None,
    random_state: int = 42,
    **kwargs,
):
    """Construct any supported model — tabular or sequence — from one call.

    Callers (:mod:`library.models.trainer`) don't need an if/else over
    model families: they call this once per requested ``mode`` and get
    back whichever kind of object that mode implies (a fitted-ready
    scikit-learn estimator, or an uninitialised ``nn.Module``).

    Parameters
    ----------
    mode : str
        A key of :data:`TABULAR_ESTIMATORS` or a member of
        :data:`SEQUENCE_MODES`.
    n_features, n_classes : int or None
        Required for sequence modes; ignored for tabular modes (sklearn
        estimators infer both from the data at ``.fit()`` time).
    random_state : int
        Forwarded to tabular estimator factories. Ignored for sequence
        modes (PyTorch initialisation isn't seeded here).
    **kwargs
        Forwarded to the underlying constructor — estimator
        hyperparameters for tabular modes, architecture hyperparameters
        (``hidden_lstm``, ``hidden_mlp``, ``n_lstm_layers``, ``dropout``)
        for sequence modes.

    Raises
    ------
    ImportError
        If a tabular mode's factory needs an optional dependency that
        isn't installed (none of the built-in modes do today, but this is
        the contract any future entry in ``TABULAR_ESTIMATORS`` should
        follow — raise, don't silently fall back).
    ValueError
        If ``mode`` isn't recognised, or a sequence mode is requested
        without ``n_features``/``n_classes``.
    """
    if mode in TABULAR_ESTIMATORS:
        return TABULAR_ESTIMATORS[mode](random_state=random_state, **kwargs)

    if mode in SEQUENCE_MODES:
        if n_features is None or n_classes is None:
            raise ValueError(
                f"Sequence mode {mode!r} requires n_features and n_classes."
            )
        return build_sequence_model(n_features, n_classes, mode=mode, **kwargs)

    raise ValueError(
        f"Unknown model mode {mode!r}. Available: "
        f"{sorted(set(TABULAR_ESTIMATORS) | SEQUENCE_MODES)}"
    )


__all__ = [
    "build_model",
    "build_sequence_model",
    "TABULAR_ESTIMATORS",
    "SEQUENCE_MODES",
]
