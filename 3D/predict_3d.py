"""
predict_3d.py — Run inference: amino-acid sequence → predicted 3-D CIF file.

Usage
-----
    # From a sequence string
    python 3D/predict_3d.py --sequence "ACDEFGHIKLMNPQRSTVWY..."

    # From a FASTA file
    python 3D/predict_3d.py --fasta my_protein.fasta

    # Validate against a reference CIF (computes Kabsch-RMSD)
    python 3D/predict_3d.py --sequence "ACE..." --reference 3D/pdb_dataset/102L.cif

Outputs (all in 3D/outputs/predictions/)
-----------------------------------------
    <name>.cif           predicted structure (Cα backbone)
    <name>_report.json   prediction metadata + validation metrics
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_loader_3d import encode_sequence, AA_TO_IDX, THREE_TO_ONE, parse_cif
from model_3d import load_checkpoint


# ── CIF writer ────────────────────────────────────────────────────────────────

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}
# Ensure all 20 standard AAs have a canonical three-letter code
_STD = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}
ONE_TO_THREE.update(_STD)


def write_cif(
    sequence: str,
    coords:   np.ndarray,   # (L, 3)  Cα positions
    out_path: str,
    entry_id: str = "PREDICTED",
) -> None:
    """Write a minimal mmCIF file containing only Cα atoms."""
    # Compute a bounding box large enough to contain the structure
    span = coords.max(axis=0) - coords.min(axis=0)
    cell_a = max(float(span[0]) + 20.0, 100.0)
    cell_b = max(float(span[1]) + 20.0, 100.0)
    cell_c = max(float(span[2]) + 20.0, 100.0)

    lines = [
        f"data_{entry_id}",
        "#",
        f"_entry.id  {entry_id}",
        "#",
        # Crystal cell (P1 triclinic — standard for single-chain models)
        f"_cell.length_a     {cell_a:.3f}",
        f"_cell.length_b     {cell_b:.3f}",
        f"_cell.length_c     {cell_c:.3f}",
        "_cell.angle_alpha  90.000",
        "_cell.angle_beta   90.000",
        "_cell.angle_gamma  90.000",
        "#",
        "_symmetry.entry_id                  " + entry_id,
        "_symmetry.space_group_name_H-M      'P 1'",
        "_symmetry.Int_Tables_number         1",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]

    for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords), start=1):
        res3 = ONE_TO_THREE.get(aa, "UNK")
        lines.append(
            f"ATOM  {i:5d}  C  CA  .  {res3}  A  1  {i:4d}  ? "
            f"{x:8.3f}  {y:8.3f}  {z:8.3f}  1.00  0.00  1"
        )

    lines.append("#")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")


# ── Kabsch RMSD ───────────────────────────────────────────────────────────────

def kabsch_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)
    H      = pred_c.T @ true_c
    U, _, Vt = np.linalg.svd(H)
    d      = np.linalg.det(Vt.T @ U.T)
    D      = np.diag([1.0, 1.0, d])
    R      = Vt.T @ D @ U.T
    diff   = pred_c @ R.T - true_c
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_sequence(
    sequence: str,
    model:    torch.nn.Module,
    max_len:  int,
    device:   str,
) -> np.ndarray:
    """
    Run model inference for a single amino-acid sequence.
    Returns Cα coordinates of shape (L, 3).
    """
    L   = min(len(sequence), max_len)
    seq = sequence[:L]

    enc, mask = encode_sequence(seq, max_len)
    enc_t  = torch.tensor(enc,  dtype=torch.long).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(enc_t, mask_t)  # (1, max_len, 3)

    coords = pred.squeeze(0).cpu().numpy()[:L]  # (L, 3)
    return coords


def load_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file. Returns list of (header, sequence) tuples."""
    entries = []
    header  = ""
    seq_buf = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if seq_buf:
                    entries.append((header, "".join(seq_buf)))
                    seq_buf = []
                header = line[1:].split()[0]
            elif line:
                seq_buf.append(line.upper())
    if seq_buf:
        entries.append((header, "".join(seq_buf)))
    return entries


# ── Validation against reference CIF ─────────────────────────────────────────

def validate(
    pred_coords: np.ndarray,
    ref_cif:     str,
    sequence:    str,
) -> dict:
    """Compare predicted structure to reference CIF via Kabsch-RMSD."""
    result = parse_cif(ref_cif)
    if result is None:
        return {"error": "Could not parse reference CIF"}

    ref_seq, ref_coords = result

    # Align by sequence length (take the shorter)
    L = min(len(pred_coords), len(ref_coords))
    if L < 4:
        return {"error": "Alignment too short"}

    rmsd = kabsch_rmsd(pred_coords[:L], ref_coords[:L])

    return {
        "reference_pdb":    Path(ref_cif).stem,
        "predicted_length": len(pred_coords),
        "reference_length": len(ref_coords),
        "aligned_length":   L,
        "kabsch_rmsd_A":    round(rmsd, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("predict_3d")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(ch)
    return logger


def main():
    ap = argparse.ArgumentParser(
        description="Predict 3-D protein structure from amino-acid sequence"
    )
    seq_group = ap.add_mutually_exclusive_group(required=True)
    seq_group.add_argument("--sequence", "-s",
                           help="One-letter amino-acid sequence string")
    seq_group.add_argument("--fasta",    "-f",
                           help="Path to input FASTA file")

    ap.add_argument("--name",      default=None,
                    help="Output file stem (default: 'predicted' or FASTA header)")
    ap.add_argument("--checkpoint",
                    default="3D/outputs/checkpoints/best_model_3d.pt",
                    help="Path to trained model checkpoint")
    ap.add_argument("--output_dir",
                    default="3D/outputs/predictions",
                    help="Directory for output files")
    ap.add_argument("--reference",  default=None,
                    help="Reference CIF file for RMSD validation")
    ap.add_argument("--max_len",    type=int, default=256)
    args = ap.parse_args()

    logger = setup_logging()

    # ── resolve paths ─────────────────────────────────────────────────────────
    root    = Path(__file__).parent.parent
    ckpt    = root / args.checkpoint
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        logger.error("Train the model first:  python 3D/train_3d.py")
        sys.exit(1)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Loading checkpoint: {ckpt}")
    model = load_checkpoint(str(ckpt), device=device)
    logger.info(f"Model loaded (device={device})")

    # ── gather sequences to predict ───────────────────────────────────────────
    if args.sequence:
        entries = [(args.name or "predicted", args.sequence.upper())]
    else:
        entries = load_fasta(str(root / args.fasta))
        if not entries:
            logger.error("No sequences found in FASTA file.")
            sys.exit(1)

    # ── predict each sequence ─────────────────────────────────────────────────
    for name, seq in entries:
        logger.info(f"\nPredicting: {name}  (length={len(seq)})")
        if len(seq) > args.max_len:
            logger.warning(
                f"  Sequence truncated from {len(seq)} to {args.max_len} residues"
            )

        t0     = time.time()
        coords = predict_sequence(seq, model, args.max_len, device)
        elapsed = time.time() - t0

        logger.info(f"  Predicted {len(coords)} Cα atoms in {elapsed:.2f}s")

        # Write CIF
        cif_out = out_dir / f"{name}.cif"
        write_cif(seq[:len(coords)], coords, str(cif_out), entry_id=name.upper())
        logger.info(f"  CIF saved → {cif_out}")

        # Validation
        val_metrics = {}
        if args.reference:
            ref_path = root / args.reference if not Path(args.reference).is_absolute() \
                       else Path(args.reference)
            logger.info(f"  Validating against: {ref_path}")
            val_metrics = validate(coords, str(ref_path), seq)
            if "kabsch_rmsd_A" in val_metrics:
                logger.info(
                    f"  Kabsch RMSD = {val_metrics['kabsch_rmsd_A']:.2f} Å  "
                    f"(aligned {val_metrics['aligned_length']} residues)"
                )
            else:
                logger.warning(f"  Validation error: {val_metrics.get('error')}")

        # Save report JSON
        report = {
            "name":           name,
            "sequence":       seq[:len(coords)],
            "length":         len(coords),
            "checkpoint":     str(ckpt),
            "device":         device,
            "inference_time_s": round(elapsed, 3),
            "output_cif":     str(cif_out),
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "validation":     val_metrics,
        }
        report_path = out_dir / f"{name}_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.info(f"  Report   → {report_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
