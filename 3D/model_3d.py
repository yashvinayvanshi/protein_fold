"""
model_3d.py — PyTorch model for 3D protein structure prediction.

Architecture
------------
  Embedding → Multi-scale CNN → BiLSTM → MLP → (x, y, z) per residue

The model predicts centroid-centered Cα coordinates for each residue.
"""

import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """Parallel 1-D convolutions at three kernel sizes, concatenated."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"
        branch = out_channels // 3
        self.conv3 = nn.Conv1d(in_channels, branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, branch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, branch, kernel_size=7, padding=3)
        self.bn    = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        out = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return F.relu(self.bn(out))  # (B, out_channels, L)


class ProteinFoldPredictor(nn.Module):
    """
    Sequence → 3-D Cα coordinate predictor.

    Input  : (B, L)         integer-encoded amino-acid sequence
    Output : (B, L, 3)      predicted Cα coordinates (centroid-centered)
    """

    def __init__(
        self,
        vocab_size:   int = 22,
        embed_dim:    int = 64,
        num_filters:  int = 192,    # must be divisible by 3
        lstm_hidden:  int = 256,
        lstm_layers:  int = 2,
        dropout:      float = 0.3,
        pad_idx:      int = 21,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.cnn = MultiScaleCNN(embed_dim, num_filters)

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out = lstm_hidden * 2  # bidirectional
        self.coord_head = nn.Sequential(
            nn.Linear(lstm_out, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # x, y, z
        )

    def forward(
        self,
        seq:  torch.Tensor,           # (B, L) long
        mask: torch.Tensor = None,    # (B, L) bool, True = valid residue
    ) -> torch.Tensor:
        """Returns predicted Cα coordinates, shape (B, L, 3)."""
        x = self.embed(seq)           # (B, L, E)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, num_filters)

        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed  = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            x, _   = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=seq.shape[1]
            )
        else:
            x, _ = self.lstm(x)  # (B, L, 2*H)

        coords = self.coord_head(x)   # (B, L, 3)
        return coords


# ── Factory helpers ───────────────────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "vocab_size":  22,
    "embed_dim":   64,
    "num_filters": 192,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "dropout":     0.3,
    "pad_idx":     21,
}


_MODEL_KEYS = set(DEFAULT_CONFIG.keys())


def build_model(config: Dict[str, Any] = None, device: str = "cpu") -> ProteinFoldPredictor:
    cfg   = {**DEFAULT_CONFIG, **(config or {})}
    model_cfg = {k: v for k, v in cfg.items() if k in _MODEL_KEYS}
    model = ProteinFoldPredictor(**model_cfg)
    return model.to(device)


def save_checkpoint(model: ProteinFoldPredictor, config: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_checkpoint(path: str, device: str = "cpu") -> ProteinFoldPredictor:
    ckpt   = torch.load(path, map_location=device)
    config = ckpt.get("config", DEFAULT_CONFIG)
    model  = build_model(config, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
