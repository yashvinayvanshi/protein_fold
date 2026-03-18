"""
data_loader.py — Data loading and preprocessing for protein secondary structure prediction.

Pipeline position: CSV → [this module] → (train_loader, val_loader, max_len)
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Amino acid vocabulary ──────────────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs
SPECIAL_TOKENS = ["*", "<PAD>"]             # '*' = masked non-standard, '<PAD>' = padding
VOCAB = AMINO_ACIDS + SPECIAL_TOKENS
AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
PAD_IDX = AA_TO_IDX["<PAD>"]               # index used as padding_idx in Embedding
VOCAB_SIZE = len(VOCAB)                     # 22

# ── Q3 secondary structure labels ─────────────────────────────────────────────
# C = Coil/Loop/Irregular  |  H = α-Helix  |  E = β-Strand
SS3_LABELS = ["C", "H", "E"]
SS3_TO_IDX = {ss: i for i, ss in enumerate(SS3_LABELS)}
IDX_TO_SS3 = {i: ss for ss, i in SS3_TO_IDX.items()}
NUM_CLASSES = len(SS3_LABELS)   # 3
PAD_LABEL = -1                  # CrossEntropyLoss ignore_index for padded positions


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(csv_path: str, filter_nonstd: bool = True) -> pd.DataFrame:
    """
    Load and optionally clean the protein dataset CSV.

    Input:
        csv_path     (str)  — path to 2018-06-06-pdb-intersect-pisces.csv
        filter_nonstd (bool) — drop rows where has_nonstd_aa == True

    Output:
        pd.DataFrame with columns [pdb_id, chain_code, seq, sst3, len]
    """
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} sequences from {csv_path}")

    if filter_nonstd:
        before = len(df)
        df = df[df["has_nonstd_aa"] == False].reset_index(drop=True)
        logging.info(f"Dropped {before - len(df)} sequences with non-standard AAs → {len(df)} remaining")

    return df[["pdb_id", "chain_code", "seq", "sst3", "len"]].copy()


# ── Encoding helpers ───────────────────────────────────────────────────────────

def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """
    Map amino acid characters to integer indices and pad/truncate to max_len.

    Input:
        seq     (str) — raw amino acid sequence, e.g. "ACDEFGHIKL"
        max_len (int) — output length (truncates if longer, pads if shorter)

    Output:
        np.ndarray of shape (max_len,), dtype int64
        Values in [0, VOCAB_SIZE-1]; unknown chars mapped to '*' index; padding = PAD_IDX
    """
    encoded = [AA_TO_IDX.get(aa, AA_TO_IDX["*"]) for aa in seq[:max_len]]
    encoded += [PAD_IDX] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)


def encode_labels(sst3: str, max_len: int) -> np.ndarray:
    """
    Map Q3 secondary structure characters to integer label indices and pad/truncate.

    Input:
        sst3    (str) — Q3 secondary structure string, e.g. "CCCHHHEEECCC"
        max_len (int) — output length

    Output:
        np.ndarray of shape (max_len,), dtype int64
        Real positions in {0, 1, 2}; padded positions = PAD_LABEL (-1, ignored in loss)
    """
    encoded = [SS3_TO_IDX.get(ss, 0) for ss in sst3[:max_len]]  # unknown → C (coil)
    encoded += [PAD_LABEL] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class ProteinDataset(Dataset):
    """
    PyTorch Dataset wrapping amino acid sequences and their Q3 secondary structure labels.

    Input  (per item):
        sequences (list[str]) — amino acid sequences
        labels    (list[str]) — Q3 secondary structure strings
        max_len   (int)       — fixed padded length

    Output (per __getitem__):
        seq_tensor   (max_len,) int64  — encoded amino acid indices
        label_tensor (max_len,) int64  — Q3 label indices (-1 for padding)
        mask_tensor  (max_len,) bool   — True for real residues, False for padding
    """

    def __init__(self, sequences: list, labels: list, max_len: int):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        sst3 = self.labels[idx]

        seq_enc = encode_sequence(seq, self.max_len)
        lbl_enc = encode_labels(sst3, self.max_len)

        # Boolean mask: True = real residue, False = padding
        real_len = min(len(seq), self.max_len)
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:real_len] = True

        return (
            torch.tensor(seq_enc, dtype=torch.long),
            torch.tensor(lbl_enc, dtype=torch.long),
            torch.tensor(mask,    dtype=torch.bool),
        )


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    max_len: int = None,
    filter_nonstd: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Full preprocessing pipeline: CSV → stratified split → PyTorch DataLoaders.

    Input:
        csv_path      (str)   — path to dataset CSV
        batch_size    (int)   — samples per mini-batch
        val_split     (float) — fraction reserved for validation (e.g. 0.1 = 10 %)
        max_len       (int)   — cap on sequence length; None → use dataset maximum
        filter_nonstd (bool)  — drop sequences containing non-standard amino acids
        seed          (int)   — random seed for reproducibility

    Output:
        train_loader (DataLoader) — shuffled training batches
        val_loader   (DataLoader) — ordered validation batches
        max_len      (int)        — resolved maximum sequence length used for padding
    """
    df = load_dataset(csv_path, filter_nonstd)

    if max_len is None:
        max_len = int(df["len"].max())
        logging.info(f"Auto-detected max_len from dataset: {max_len}")

    sequences = df["seq"].tolist()
    labels    = df["sst3"].tolist()

    train_seqs, val_seqs, train_lbls, val_lbls = train_test_split(
        sequences, labels, test_size=val_split, random_state=seed
    )
    logging.info(f"Split → Train: {len(train_seqs)} | Val: {len(val_seqs)} sequences")

    train_ds = ProteinDataset(train_seqs, train_lbls, max_len)
    val_ds   = ProteinDataset(val_seqs,   val_lbls,   max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, max_len
