"""
data_loader_3d.py — Parse CIF files, extract sequences and Cα coordinates,
build PyTorch Dataset/DataLoader for 3D structure prediction training.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Amino-acid vocabulary ─────────────────────────────────────────────────────
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine → methionine
    "HSD": "H", "HSE": "H", "HSP": "H",  # histidine variants
    "GLX": "E", "ASX": "N",
}
AA_LIST   = list("ACDEFGHIKLMNPQRSTVWY") + ["X", "<PAD>"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
VOCAB_SIZE = len(AA_LIST)          # 22
PAD_IDX    = AA_TO_IDX["<PAD>"]   # 21


# ── CIF parser ────────────────────────────────────────────────────────────────

def _find_atom_site_block(lines: List[str]):
    """
    Scan for the loop_ block whose columns begin with _atom_site.
    Returns (col_map, data_lines) or (None, None).
    col_map: dict mapping '_atom_site.<name>' → column index (0-based)
    data_lines: list of raw text lines (each a data row)
    """
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            # collect column headers
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1

            if not any(c.startswith("_atom_site.") for c in cols):
                i = j
                continue

            col_map = {c: idx for idx, c in enumerate(cols)}

            # collect data rows until a non-data line
            data = []
            while j < len(lines):
                raw = lines[j].rstrip()
                s   = raw.strip()
                if not s or s == "loop_" or s.startswith("_") or s.startswith("#"):
                    break
                data.append(s)
                j += 1

            return col_map, data
        i += 1

    return None, None


def parse_cif(cif_path: str) -> Optional[Tuple[str, np.ndarray]]:
    """
    Parse a mmCIF file and extract:
      - the one-letter amino-acid sequence of the first protein chain
      - Cα (alpha-carbon) 3-D coordinates as a float32 array of shape (L, 3)

    Returns None if the file cannot be parsed or is too short.
    """
    try:
        with open(cif_path, "r", errors="ignore") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    col_map, data_lines = _find_atom_site_block(lines)
    if col_map is None or not data_lines:
        return None

    # Required columns
    required = {
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    }
    if not required.issubset(col_map):
        return None

    gi  = col_map["_atom_site.group_PDB"]
    ai  = col_map["_atom_site.label_atom_id"]
    ci  = col_map["_atom_site.label_comp_id"]
    chi = col_map["_atom_site.label_asym_id"]
    si  = col_map["_atom_site.label_seq_id"]
    xi  = col_map["_atom_site.Cartn_x"]
    yi  = col_map["_atom_site.Cartn_y"]
    zi  = col_map["_atom_site.Cartn_z"]

    # Optional: model number column
    mi = col_map.get("_atom_site.pdbx_PDB_model_num")

    residues: Dict[int, Tuple[str, float, float, float]] = {}
    first_chain: Optional[str] = None
    min_cols = max(gi, ai, ci, chi, si, xi, yi, zi) + 1

    for row in data_lines:
        parts = row.split()
        if len(parts) < min_cols:
            continue
        if parts[gi] != "ATOM":
            continue
        if parts[ai] != "CA":
            continue
        # Only first model
        if mi is not None and len(parts) > mi:
            try:
                if int(parts[mi]) != 1:
                    continue
            except ValueError:
                pass

        chain = parts[chi]
        if first_chain is None:
            first_chain = chain
        if chain != first_chain:
            continue

        try:
            seq_id = int(parts[si])
            aa3    = parts[ci]
            x      = float(parts[xi])
            y      = float(parts[yi])
            z      = float(parts[zi])
        except (ValueError, IndexError):
            continue

        if seq_id not in residues:
            residues[seq_id] = (aa3, x, y, z)

    if len(residues) < 20:
        return None

    sorted_res = sorted(residues.items())
    sequence = ""
    coords   = []
    for _, (aa3, x, y, z) in sorted_res:
        sequence += THREE_TO_ONE.get(aa3, "X")
        coords.append([x, y, z])

    if len(sequence) < 20:
        return None

    return sequence, np.array(coords, dtype=np.float32)


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_sequence(seq: str, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (encoded indices, bool mask), both length max_len."""
    enc  = np.full(max_len, PAD_IDX, dtype=np.int64)
    mask = np.zeros(max_len, dtype=bool)
    for i, aa in enumerate(seq[:max_len]):
        enc[i]  = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
        mask[i] = True
    return enc, mask


def center_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract centroid. Returns (centered_coords, centroid)."""
    centroid = coords.mean(axis=0)
    return coords - centroid, centroid


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProteinCoordsDataset(Dataset):
    """
    Each item: dict with keys
      'pdb_id'   : str
      'sequence' : str (original 1-letter sequence)
      'seq_len'  : int
      'encoded'  : LongTensor  (max_len,)
      'coords'   : FloatTensor (max_len, 3)  — centered Cα coords, padded with 0
      'mask'     : BoolTensor  (max_len,)
    """

    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "pdb_id":   r["pdb_id"],
            "sequence": r["sequence"],
            "seq_len":  r["seq_len"],
            "encoded":  torch.tensor(r["encoded"], dtype=torch.long),
            "coords":   torch.tensor(r["coords"],  dtype=torch.float32),
            "mask":     torch.tensor(r["mask"],     dtype=torch.bool),
        }


def load_dataset(
    cif_dir:   str,
    max_len:   int  = 256,
    max_files: int  = None,
    logger:    logging.Logger = None,
) -> List[Dict]:
    """
    Parse all CIF files in cif_dir and return a list of record dicts.
    Coordinates are centered (centroid removed) per protein.
    """
    cif_dir   = Path(cif_dir)
    cif_files = sorted(cif_dir.glob("*.cif"))
    if max_files:
        cif_files = cif_files[:max_files]

    total   = len(cif_files)
    records = []
    failed  = 0

    _log = logger.info if logger else print

    for k, cif_path in enumerate(cif_files, 1):
        if k % 100 == 0 or k == total:
            _log(f"  Parsing CIF files … {k}/{total}")

        result = parse_cif(str(cif_path))
        if result is None:
            failed += 1
            continue

        seq, coords = result

        # Truncate to max_len
        if len(seq) > max_len:
            seq    = seq[:max_len]
            coords = coords[:max_len]

        L           = len(seq)
        enc, mask   = encode_sequence(seq, max_len)

        # Center coordinates (translation-invariant representation)
        centered, _ = center_coords(coords)

        padded_coords = np.zeros((max_len, 3), dtype=np.float32)
        padded_coords[:L] = centered

        records.append({
            "pdb_id":   cif_path.stem,
            "sequence": seq,
            "seq_len":  L,
            "encoded":  enc,
            "coords":   padded_coords,
            "mask":     mask,
        })

    _log(f"Dataset: {len(records)} proteins loaded, {failed} skipped.")
    return records


def get_dataloaders(
    cif_dir:    str,
    max_len:    int   = 256,
    batch_size: int   = 8,
    val_frac:   float = 0.10,
    max_files:  int   = None,
    num_workers: int  = 0,
    logger:     logging.Logger = None,
) -> Tuple[DataLoader, DataLoader, List[Dict]]:
    """
    Load dataset, split train/val, return (train_loader, val_loader, all_records).
    """
    records = load_dataset(cif_dir, max_len=max_len, max_files=max_files, logger=logger)

    dataset   = ProteinCoordsDataset(records)
    n_val     = max(1, int(len(dataset) * val_frac))
    n_train   = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, records
