"""
model.py — CNN + BiLSTM model for per-residue secondary structure prediction (Q3).

Architecture:
    Embedding → Multi-scale CNN → Dropout → BiLSTM → Linear Classifier

Pipeline position: [data_loader] → [this module] → per-residue logits
"""

import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    """
    Parallel 1D convolutions at multiple kernel sizes to capture local sequence patterns
    (short-range: 3-residue windows; medium: 5; long-range local: 7).

    Input:
        x (batch, seq_len, embed_dim) — embedded amino acid sequence

    Output:
        (batch, seq_len, num_filters * len(kernel_sizes)) — concatenated feature maps
    """

    def __init__(self, embed_dim: int, num_filters: int = 64, kernel_sizes: tuple = (3, 5, 7)):
        super().__init__()
        # Each kernel size gets its own Conv1d → ReLU → BatchNorm block
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
            )
            for k in kernel_sizes
        ])
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (batch, channels, seq_len); transpose in and out
        x_t = x.transpose(1, 2)                                        # (batch, embed_dim, seq_len)
        features = [conv(x_t).transpose(1, 2) for conv in self.convs]  # each: (batch, seq_len, num_filters)
        return torch.cat(features, dim=-1)                              # (batch, seq_len, out_dim)


class ProteinSSPredictor(nn.Module):
    """
    Full protein secondary structure predictor.

    Forward pass:
        1. Embedding   : amino acid indices → dense vectors
        2. MultiScaleCNN: local pattern extraction at kernel sizes 3, 5, 7
        3. Dropout     : regularisation
        4. BiLSTM      : long-range bidirectional context
        5. Classifier  : Linear → ReLU → Dropout → Linear per residue

    Input:
        x (batch, seq_len) int64 — integer-encoded amino acid indices

    Output:
        logits (batch, seq_len, num_classes) float — un-normalised class scores per residue
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 5, 7),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 3,
        dropout: float   = 0.3,
        pad_idx: int     = 21,
    ):
        super().__init__()

        # Embedding: token index → dense vector; PAD_IDX embeddings are kept at zero
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Multi-scale CNN for local residue context
        self.cnn = MultiScaleCNN(embed_dim, num_filters, kernel_sizes)
        cnn_out_dim = self.cnn.out_dim          # num_filters * len(kernel_sizes)

        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM for global sequence context
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Per-residue classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),   # BiLSTM doubles hidden size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x      (batch, seq_len) int64
        Output: logits (batch, seq_len, num_classes) float
        """
        emb      = self.embedding(x)            # (batch, seq_len, embed_dim)
        cnn_out  = self.cnn(emb)                # (batch, seq_len, cnn_out_dim)
        cnn_out  = self.dropout(cnn_out)
        lstm_out, _ = self.bilstm(cnn_out)      # (batch, seq_len, lstm_hidden * 2)
        logits   = self.classifier(lstm_out)    # (batch, seq_len, num_classes)
        return logits


def build_model(config: dict, device: torch.device) -> "ProteinSSPredictor":
    """
    Convenience factory: instantiate ProteinSSPredictor from a config dict and move to device.

    Input:
        config (dict) — keys: vocab_size, embed_dim, num_filters, lstm_hidden,
                               lstm_layers, num_classes, dropout, pad_idx
        device (torch.device)

    Output:
        ProteinSSPredictor on the specified device
    """
    model = ProteinSSPredictor(
        vocab_size  = config["vocab_size"],
        embed_dim   = config["embed_dim"],
        num_filters = config["num_filters"],
        lstm_hidden = config["lstm_hidden"],
        lstm_layers = config["lstm_layers"],
        num_classes = config["num_classes"],
        dropout     = config["dropout"],
        pad_idx     = config["pad_idx"],
    ).to(device)
    return model
