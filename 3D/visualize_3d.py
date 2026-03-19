"""
visualize_3d.py — Visualise a protein Cα backbone from a CIF file and save
the result as a PNG image.

Usage
-----
    # Visualise a single CIF
    python 3D/visualize_3d.py --cif 3D/outputs/predictions/predicted.cif

    # Overlay predicted vs reference
    python 3D/visualize_3d.py \
        --cif       3D/outputs/predictions/predicted.cif \
        --reference 3D/pdb_dataset/102L.cif \
        --out       3D/outputs/predictions/overlay.png

    # Visualise from a raw sequence string (uses saved checkpoint)
    python 3D/visualize_3d.py --sequence "ACDEFGHIKLMNPQRSTVWY..."
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_loader_3d import parse_cif, THREE_TO_ONE


# ── CIF reader (wraps the shared parser) ─────────────────────────────────────

def load_backbone(cif_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Return (sequence, coords) from a CIF file, or (None, None)."""
    result = parse_cif(cif_path)
    if result is None:
        return None, None
    return result          # (seq_str, np.ndarray (L,3))


# ── Colour helpers ────────────────────────────────────────────────────────────

def _residue_colours(n: int, cmap_name: str = "plasma") -> np.ndarray:
    """N rainbow colours from N-terminal (cool) → C-terminal (warm)."""
    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0.05, 0.95, n))


# ── Core visualisation ────────────────────────────────────────────────────────

def _add_backbone_3d(ax, coords: np.ndarray, label: str, colours,
                     alpha: float = 1.0, linewidth: float = 1.5):
    """Draw Cα backbone as a coloured 3-D line on *ax*."""
    segments = [
        [coords[i], coords[i + 1]]
        for i in range(len(coords) - 1)
    ]
    lc = Line3DCollection(
        segments,
        colors=colours[:-1],
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    ax.add_collection3d(lc)

    # Mark N-terminus and C-terminus
    ax.scatter(*coords[0],  s=60, color=colours[0],   zorder=5, marker="o")
    ax.scatter(*coords[-1], s=60, color=colours[-1],  zorder=5, marker="s")


def visualise_single(
    cif_path:    str,
    out_png:     str,
    title:       str = None,
) -> None:
    """Render one CIF backbone and save to PNG."""
    seq, coords = load_backbone(cif_path)
    if coords is None:
        raise ValueError(f"Could not parse CIF: {cif_path}")

    cols = _residue_colours(len(coords))
    fig  = plt.figure(figsize=(8, 7))
    ax   = fig.add_subplot(111, projection="3d")

    _add_backbone_3d(ax, coords, label=Path(cif_path).stem, colours=cols)

    # Axis limits
    mx, mn = coords.max(axis=0), coords.min(axis=0)
    pad    = (mx - mn).max() * 0.1 + 2
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mn[1] - pad, mx[1] + pad)
    ax.set_zlim(mn[2] - pad, mx[2] + pad)

    ax.set_xlabel("X (Å)", fontsize=9)
    ax.set_ylabel("Y (Å)", fontsize=9)
    ax.set_zlabel("Z (Å)", fontsize=9)
    ax.set_title(title or f"{Path(cif_path).stem}  |  {len(coords)} residues",
                 fontsize=11, pad=12)

    # Colour-bar: N→C
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=1, vmax=len(coords)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, aspect=20)
    cb.set_label("Residue index (N→C)", fontsize=8)

    ax.scatter([], [], s=60, color="k", marker="o", label="N-terminus")
    ax.scatter([], [], s=60, color="k", marker="s", label="C-terminus")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def visualise_overlay(
    pred_cif:   str,
    ref_cif:    str,
    out_png:    str,
    title:      str = None,
) -> None:
    """Render predicted (colour) and reference (grey) backbones overlaid."""
    seq_p, coords_p = load_backbone(pred_cif)
    seq_r, coords_r = load_backbone(ref_cif)

    if coords_p is None:
        raise ValueError(f"Could not parse predicted CIF: {pred_cif}")
    if coords_r is None:
        raise ValueError(f"Could not parse reference CIF: {ref_cif}")

    # Superpose reference on predicted (Kabsch)
    def kabsch(A, B):
        Ac = A - A.mean(0); Bc = B - B.mean(0)
        H  = Ac.T @ Bc
        U, _, Vt = np.linalg.svd(H)
        d  = np.linalg.det(Vt.T @ U.T)
        D  = np.diag([1., 1., d])
        R  = Vt.T @ D @ U.T
        return Ac @ R.T, Bc

    L     = min(len(coords_p), len(coords_r))
    cp, cr = kabsch(coords_p[:L], coords_r[:L])

    cols_pred = _residue_colours(L, cmap_name="plasma")
    cols_ref  = np.tile([0.55, 0.55, 0.55, 0.5], (L, 1))

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    _add_backbone_3d(ax, cr, label=f"Reference ({Path(ref_cif).stem})",
                     colours=cols_ref, alpha=0.5, linewidth=1.2)
    _add_backbone_3d(ax, cp, label=f"Predicted ({Path(pred_cif).stem})",
                     colours=cols_pred, alpha=0.9, linewidth=2.0)

    all_pts = np.vstack([cp, cr])
    mx, mn  = all_pts.max(0), all_pts.min(0)
    pad     = (mx - mn).max() * 0.1 + 2
    ax.set_xlim(mn[0]-pad, mx[0]+pad)
    ax.set_ylim(mn[1]-pad, mx[1]+pad)
    ax.set_zlim(mn[2]-pad, mx[2]+pad)

    ax.set_xlabel("X (Å)", fontsize=9)
    ax.set_ylabel("Y (Å)", fontsize=9)
    ax.set_zlabel("Z (Å)", fontsize=9)

    # RMSD annotation
    diff  = cp - cr
    rmsd  = float(np.sqrt((diff**2).sum(1).mean()))
    stitle = title or "Predicted vs Reference"
    ax.set_title(f"{stitle}  |  L={L}  RMSD={rmsd:.2f} Å", fontsize=11, pad=12)

    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=1, vmax=L))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, aspect=20)
    cb.set_label("Residue index (N→C)", fontsize=8)

    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def visualise_2d_projections(
    cif_path: str,
    out_png:  str,
    title:    str = None,
) -> None:
    """
    Three 2-D projections (XY, XZ, YZ) of the Cα backbone — useful as a
    quick sanity check without requiring a 3-D renderer.
    """
    seq, coords = load_backbone(cif_path)
    if coords is None:
        raise ValueError(f"Could not parse CIF: {cif_path}")

    cols = _residue_colours(len(coords))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    projections = [
        (0, 1, "X (Å)", "Y (Å)", "XY plane"),
        (0, 2, "X (Å)", "Z (Å)", "XZ plane"),
        (1, 2, "Y (Å)", "Z (Å)", "YZ plane"),
    ]

    for ax, (xi, yi, xl, yl, pname) in zip(axes, projections):
        segs = [
            [[coords[i, xi], coords[i, yi]], [coords[i+1, xi], coords[i+1, yi]]]
            for i in range(len(coords) - 1)
        ]
        lc = LineCollection(segs, colors=cols[:-1], linewidth=1.5)
        ax.add_collection(lc)
        ax.scatter(coords[0,  xi], coords[0,  yi], s=40, color="blue",  zorder=5, label="N")
        ax.scatter(coords[-1, xi], coords[-1, yi], s=40, color="red",   zorder=5, label="C")
        ax.autoscale()
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(pname, fontsize=10)
        ax.legend(fontsize=8)

    stem = Path(cif_path).stem
    fig.suptitle(title or f"{stem}  |  {len(coords)} residues",
                 fontsize=12, y=1.02)

    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(1, len(coords)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.8, label="Residue index (N→C)")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise protein Cα backbone from CIF file(s)"
    )
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--cif",      help="Path to input CIF file to visualise")
    inp.add_argument("--sequence", help="Amino-acid sequence (runs inference first)")

    ap.add_argument("--reference",  default=None,
                    help="Reference CIF for overlay comparison")
    ap.add_argument("--out",        default=None,
                    help="Output PNG path (auto-generated if omitted)")
    ap.add_argument("--title",      default=None)
    ap.add_argument("--projections", action="store_true",
                    help="Also save 2-D projection panel")
    ap.add_argument("--checkpoint",
                    default="3D/outputs/checkpoints/best_model_3d.pt",
                    help="Model checkpoint (used when --sequence is given)")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    out_dir = ROOT / "3D" / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── if sequence is given, run inference first ─────────────────────────────
    if args.sequence:
        import torch
        from data_loader_3d import encode_sequence
        from model_3d import load_checkpoint
        from predict_3d import predict_sequence, write_cif

        ckpt_path = ROOT / args.checkpoint
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            print("Train the model first:  python 3D/train_3d.py")
            sys.exit(1)

        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        model  = load_checkpoint(str(ckpt_path), device=device)
        seq    = args.sequence.upper()
        coords = predict_sequence(seq, model, args.max_len, device)

        cif_path = str(out_dir / "predicted_from_sequence.cif")
        write_cif(seq[:len(coords)], coords, cif_path)
        args.cif = cif_path
        print(f"  Predicted CIF → {cif_path}")

    # ── visualise ─────────────────────────────────────────────────────────────
    stem    = Path(args.cif).stem
    out_png = args.out or str(out_dir / f"{stem}_3d.png")

    if args.reference:
        print(f"Rendering overlay: {stem} vs {Path(args.reference).stem}")
        visualise_overlay(args.cif, args.reference, out_png, title=args.title)
    else:
        print(f"Rendering backbone: {stem}")
        visualise_single(args.cif, out_png, title=args.title)

    if args.projections:
        proj_png = str(Path(out_png).with_name(
            Path(out_png).stem + "_projections.png"
        ))
        visualise_2d_projections(args.cif, proj_png, title=args.title)

    print("Done.")


if __name__ == "__main__":
    main()
