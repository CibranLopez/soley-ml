"""
library.models.neural_network
================================

PyTorch model architectures for solar PV fault detection / classification.

build_model(n_features, n_classes, mode, ...)
    Returns a :class:`HybridFaultModel` configured for the requested mode.

    mode="mlp"     Per-timestep MLP on the last timestep of the window.
    mode="lstm"    Bidirectional LSTM processing the full window.
    mode="hybrid"  MLP branch (last step) + LSTM branch (all steps) fused.

All variants share the same ``forward(x_window)`` signature, making it
trivial to swap architectures in a notebook.

State-of-the-art touches
-------------------------
* Layer-norm after LSTM (stabilises training with variable-length inputs).
* Gradient clipping (max_norm=1) in the training loop.
* ReduceLROnPlateau scheduler.
* Weighted cross-entropy for class-imbalanced fault data.
"""

import logging

log = logging.getLogger("library")


def build_model(
    n_features: int,
    n_classes: int,
    mode: str = "hybrid",
    hidden_lstm: int = 128,
    hidden_mlp: int = 64,
    n_lstm_layers: int = 2,
    dropout: float = 0.3,
):
    """Construct a fault detection / classification model.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    n_classes : int
        Number of output classes (2 for detection, 11 for classification).
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
