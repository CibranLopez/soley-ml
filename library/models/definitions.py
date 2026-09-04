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
    ``{"mlp", "lstm", "hybrid", "transformer"}`` — the PyTorch
    architectures, all built by the single ``HybridFaultModel`` class
    gated on ``mode``.

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
                         min_samples_leaf=20, n_jobs=-1,
                         class_weight="balanced", random_state=42, **extra):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
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

SEQUENCE_MODES = {"mlp", "lstm", "hybrid", "transformer"}


def build_sequence_model(
    n_features: int,
    n_classes: int,
    mode: str = "hybrid",
    hidden_lstm: int = 128,
    hidden_mlp: int = 64,
    n_lstm_layers: int = 2,
    dropout: float = 0.3,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    transformer_ff_dim: int = 128,
):
    """Construct a fault detection / classification sequence model.

    mode="mlp"          Per-timestep MLP on the last timestep of the window.
    mode="lstm"          Unidirectional LSTM processing the full window.
    mode="hybrid"        MLP branch (last step) + LSTM branch (all steps) fused.
    mode="transformer"   Self-attention over the full window (all steps) —
                         an alternative to the LSTM branch, not a fusion with
                         it. See "Why not a literal port" below.

    All four variants share the same ``forward(x_window)`` signature.

    Why not a literal port from the portfolio-RL project
    ------------------------------------------------------
    This started as a request to reuse ``attention_broad_pairwise_
    transformer`` from a different (RL/portfolio) project wholesale.
    That's not possible as a copy — it's built on
    ``stable_baselines3``/``gym`` RL scaffolding this project doesn't have,
    and its attention axis is fundamentally different: its
    ``MultiHeadAttentionExtractor`` runs self-attention ACROSS ASSETS
    observed in parallel at one timestep ("how should AAPL's allocation
    depend on what NVDA is doing right now") — there's no "N systems
    observed in parallel" axis in a Soley-ml sample; each window is one
    system's own sensor history over time. What *is* portable is the
    architectural idea (real query/key/value self-attention via
    ``nn.TransformerEncoderLayer``, instead of a single shared-gate MLP),
    re-pointed at the axis that actually exists here: TIME. This mode runs
    self-attention across the window's timesteps, as a genuine alternative
    to ``lstm``'s recurrence over the same axis.

    One consequence of attending across time instead of across assets:
    self-attention has no inherent sense of order (permutation-invariant)
    — the ported source didn't need positional encoding because asset
    order is arbitrary (which ticker is "first" doesn't matter), but here
    timestep order is exactly the signal a fault-progression pattern is
    made of. This mode adds a sinusoidal positional encoding
    (Vaswani et al. 2017) — computed fresh each forward pass from the
    window's actual length, not a fixed/learned embedding, so it needs no
    extra constructor argument and imposes no maximum window size — as a
    required correction for this axis change, not an optional add-on.

    A second, smaller adaptation: the ported extractor mean-pools its
    attended output across assets (there's no natural "last" asset).
    Here, to keep this mode a faithful, comparable swap-in for what the
    ``lstm`` branch already contributes to ``hybrid`` (its last hidden
    state — "the summary of everything up to now, evaluated now"), this
    mode reads out the LAST timestep's attended representation instead of
    mean-pooling the whole window. That's a deliberate choice, not an
    oversight — mean-pooling across time is a defensible alternative if
    you'd rather it summarize the whole window symmetrically.

    Not validated against real PyTorch in this environment (no torch
    available here — same limitation the ported source's own docstring
    already notes for itself). Built entirely from PyTorch's standard
    ``nn.TransformerEncoderLayer``/``nn.TransformerEncoder`` rather than a
    hand-rolled attention formula, and the shape bookkeeping was verified
    with a pure-numpy shape simulation (see this project's test suite).
    Sanity-check ``model(sample_window).shape == (batch, n_classes)`` on
    your first real run before trusting it.

    State-of-the-art touches
    -------------------------
    * Layer-norm after LSTM / after transformer pooling (stabilises
      training with variable-length inputs).
    * Gradient clipping (max_norm=1) in the training loop.
    * ReduceLROnPlateau scheduler.
    * Weighted cross-entropy for class-imbalanced fault data.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_classes : int
        Number of output classes (2 for detection, N for classification).
    mode : {"mlp", "lstm", "hybrid", "transformer"}
    hidden_lstm : int
        LSTM hidden size. Ignored for mode="transformer".
    hidden_mlp : int
        MLP hidden size (per layer).
    n_lstm_layers : int
        Ignored for mode="transformer".
    dropout : float
        Also used inside the transformer encoder layers when
        mode="transformer".
    embed_dim : int
        Transformer token dimension (per-timestep encoding size). Must be
        divisible by ``num_heads``. Only used for mode="transformer".
    num_heads : int
        Self-attention heads. Only used for mode="transformer".
    num_transformer_layers : int
        Stacked ``nn.TransformerEncoderLayer`` count. Only used for
        mode="transformer".
    transformer_ff_dim : int
        Feed-forward width inside each transformer encoder layer. Only
        used for mode="transformer".

    Returns
    -------
    torch.nn.Module
    """
    import torch
    import torch.nn as nn

    if mode == "transformer" and embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

    def _sinusoidal_positional_encoding(seq_len: int, dim: int, device, dtype):
        """Standard fixed (non-learned) sin/cos positional encoding
        (Vaswani et al. 2017), computed fresh for the actual window length
        seen at forward time — see the "Why not a literal port" note
        above for why this exists at all (the ported source's attention
        axis — assets — didn't need one; time order does)."""
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim)
        )
        pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe  # (seq_len, dim) — broadcasts over the batch dimension

    class HybridFaultModel(nn.Module):
        """MLP / LSTM / Hybrid / Transformer model for PV fault analysis."""

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

            if mode == "transformer":
                # Per-timestep token encoder: n_features -> embed_dim,
                # mirroring how the ported source's asset_encoder projects
                # each asset's raw features before attention.
                self.token_encoder = nn.Sequential(
                    nn.Linear(n_features, embed_dim),
                    nn.ReLU(),
                    nn.LayerNorm(embed_dim),
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=dropout, batch_first=True, activation="relu",
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_transformer_layers
                )
                self.transformer_norm = nn.LayerNorm(embed_dim)

            head_in = 0
            if mode in ("mlp", "hybrid"):
                head_in += hidden_mlp
            if mode in ("lstm", "hybrid"):
                head_in += hidden_lstm
            if mode == "transformer":
                head_in += embed_dim

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
            if self.mode == "transformer":
                tokens = self.token_encoder(x_window)      # (batch, window, embed_dim)
                seq_len = tokens.shape[1]
                pos = _sinusoidal_positional_encoding(
                    seq_len, embed_dim, tokens.device, tokens.dtype
                )
                tokens = tokens + pos.unsqueeze(0)          # broadcast over batch
                attended = self.transformer(tokens)          # (batch, window, embed_dim)
                attended_last = attended[:, -1, :]            # last timestep, post-attention
                parts.append(self.transformer_norm(attended_last))
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
        (``hidden_lstm``, ``hidden_mlp``, ``n_lstm_layers``, ``dropout``,
        plus ``embed_dim``/``num_heads``/``num_transformer_layers``/
        ``transformer_ff_dim`` for mode="transformer" — see
        :func:`build_sequence_model`) for sequence modes.

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
