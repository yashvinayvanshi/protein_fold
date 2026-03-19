"""
render_cif.py — Render a protein CIF file using py3Dmol (HTML) and
                matplotlib (PNG).

Usage
-----
    python3 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif
    python3 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif \
                             --style cartoon
    python3 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif \
                             --style sphere --color spectrum

Outputs
-------
    <stem>_viewer.html   — interactive 3-D viewer (open in any browser)
    <stem>_render.png    — static high-quality PNG (matplotlib tube plot)

py3Dmol style options : cartoon | stick | sphere | line | cross
py3Dmol color options : spectrum | chainHetatm | ssJmol | default
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import py3Dmol

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from data_loader_3d import parse_cif


# ── py3Dmol → HTML ────────────────────────────────────────────────────────────

def render_html(cif_path: str, out_html: str,
                style: str = "cartoon", color: str = "spectrum",
                width: int = 900, height: int = 700) -> None:
    """
    Generate a self-contained HTML file with an interactive 3-D molecular
    viewer powered by py3Dmol / 3Dmol.js.
    """
    cif_text = Path(cif_path).read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_text, "cif")

    style_dict: dict
    if style == "cartoon":
        style_dict = {"cartoon": {"color": color, "thickness": 0.4}}
    elif style == "stick":
        style_dict = {"stick": {"colorscheme": color}}
    elif style == "sphere":
        style_dict = {"sphere": {"color": color, "scale": 0.4}}
    elif style == "line":
        style_dict = {"line": {"color": color}}
    else:
        style_dict = {"cross": {"color": color}}

    view.setStyle({}, style_dict)
    view.setBackgroundColor("white")
    view.zoomTo()

    # py3Dmol generates an HTML string with embedded JS
    html_body = view._make_html()

    # Wrap in a minimal page with title + download note
    stem  = Path(cif_path).stem
    n_res = len(parse_cif(cif_path)[0]) if parse_cif(cif_path) else "?"
    html  = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Protein 3D Viewer — {stem}</title>
  <style>
    body {{ font-family: sans-serif; background: #f5f5f5; margin: 0; }}
    h2   {{ text-align:center; padding:10px; color:#333; }}
    p    {{ text-align:center; font-size:13px; color:#666; margin:0 0 6px; }}
    .viewer-wrap {{ display:flex; justify-content:center; padding:10px; }}
  </style>
</head>
<body>
  <h2>Predicted Structure — {stem}</h2>
  <p>{n_res} residues &nbsp;|&nbsp; Style: {style} &nbsp;|&nbsp;
     Color: {color} &nbsp;|&nbsp; Left-drag: rotate &nbsp;|&nbsp;
     Scroll: zoom &nbsp;|&nbsp; Right-drag: translate</p>
  <div class="viewer-wrap">
    {html_body}
  </div>
</body>
</html>"""

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html)
    print(f"  HTML viewer → {out_html}")


# ── matplotlib tube-style PNG ─────────────────────────────────────────────────

def _tube_segments(coords: np.ndarray, colours: np.ndarray,
                   widths: np.ndarray) -> list:
    """Build per-segment colour + linewidth lists for Line3DCollection."""
    segs, cols, lws = [], [], []
    for i in range(len(coords) - 1):
        segs.append([coords[i], coords[i + 1]])
        cols.append(colours[i])
        lws.append(widths[i])
    return segs, cols, lws


def render_png(cif_path: str, out_png: str, dpi: int = 200) -> None:
    """
    High-quality matplotlib tube-plot:
      - Wide backbone tubes coloured N→C (plasma)
      - Thin shadow pass for depth perception
      - Spheres at Cα positions sized by local curvature
      - Clean dark background
    """
    result = parse_cif(cif_path)
    if result is None:
        print(f"  ERROR: Could not parse {cif_path}")
        return
    seq, coords = result
    L = len(coords)

    cmap = plt.get_cmap("plasma")
    t    = np.linspace(0.05, 0.95, L)
    cols = cmap(t)

    # Local curvature → sphere size
    curvature = np.ones(L) * 30
    if L > 2:
        d1 = np.diff(coords, axis=0)
        d2 = np.diff(d1,      axis=0)
        k  = np.linalg.norm(d2, axis=1)
        k  = (k - k.min()) / (k.max() - k.min() + 1e-8)
        curvature[1:-1] = 20 + k * 60

    # Linewidths: thicker in centre, thinner at termini
    lw_base = np.ones(L) * 2.2
    fade    = np.linspace(0.5, 1.0, L // 2)
    fade    = np.concatenate([fade, fade[::-1]]) if L % 2 == 0 \
              else np.concatenate([fade, [1.0], fade[::-1]])
    lw_base = lw_base * fade[:L]

    # Build figure
    fig = plt.figure(figsize=(9, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Shadow pass (offset, grey, thin)
    shadow = coords.copy()
    shadow[:, 2] = coords[:, 2].min() - 3   # project onto floor
    segs_s, _, _ = _tube_segments(shadow, cols, lw_base)
    lc_s = Line3DCollection(segs_s, colors=[(0.2, 0.2, 0.2, 0.3)] * len(segs_s),
                             linewidth=0.8, zorder=1)
    ax.add_collection3d(lc_s)

    # Main backbone tubes
    segs, _, lws = _tube_segments(coords, cols, lw_base)
    lc = Line3DCollection(segs, colors=cols[:-1], linewidths=lws, zorder=3,
                          capstyle="round", joinstyle="round")
    ax.add_collection3d(lc)

    # Cα spheres
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=t, cmap="plasma", s=curvature,
               alpha=0.85, zorder=4, edgecolors="none", depthshade=True)

    # N-terminus (blue) and C-terminus (red) markers
    ax.scatter(*coords[0],  s=120, color="#00d4ff", zorder=6,
               edgecolors="white", linewidths=0.5, label="N-terminus")
    ax.scatter(*coords[-1], s=120, color="#ff6b6b", zorder=6,
               edgecolors="white", linewidths=0.5, label="C-terminus")

    # Axis limits
    mx, mn = coords.max(0), coords.min(0)
    pad    = (mx - mn).max() * 0.12 + 3
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mn[1] - pad, mx[1] + pad)
    ax.set_zlim(mn[2] - pad, mx[2] + pad)

    # Style
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=7)
    ax.xaxis.label.set_color("#aaaacc"); ax.set_xlabel("X (Å)", fontsize=8)
    ax.yaxis.label.set_color("#aaaacc"); ax.set_ylabel("Y (Å)", fontsize=8)
    ax.zaxis.label.set_color("#aaaacc"); ax.set_zlabel("Z (Å)", fontsize=8)
    ax.grid(True, color="#333355", linewidth=0.4, alpha=0.5)

    # Colour-bar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=Normalize(1, L))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=22)
    cb.set_label("Residue index (N→C)", color="#ccccee", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#ccccee", labelcolor="#ccccee", labelsize=7)

    stem = Path(cif_path).stem
    ax.set_title(f"{stem}  —  {L} residues",
                 color="#e0e0ff", fontsize=12, pad=14)
    ax.legend(fontsize=8, loc="upper left",
              facecolor="#22224a", edgecolor="#555577",
              labelcolor="#ddddff")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG render   → {out_png}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Render a protein CIF — HTML (py3Dmol) + PNG (matplotlib)"
    )
    ap.add_argument("--cif",    required=True, help="Input CIF file")
    ap.add_argument("--outdir", default=None,  help="Output directory (default: same as CIF)")
    ap.add_argument("--style",  default="cartoon",
                    choices=["cartoon", "stick", "sphere", "line", "cross"])
    ap.add_argument("--color",  default="spectrum",
                    help="py3Dmol colour scheme (spectrum / chainHetatm / ssJmol)")
    ap.add_argument("--dpi",    type=int, default=200)
    args = ap.parse_args()

    cif_path = Path(ROOT / args.cif) if not Path(args.cif).is_absolute() \
               else Path(args.cif)

    if not cif_path.exists():
        print(f"ERROR: CIF not found: {cif_path}")
        sys.exit(1)

    out_dir = Path(args.outdir) if args.outdir else cif_path.parent
    stem    = cif_path.stem

    print(f"\nRendering: {cif_path.name}")
    render_html(str(cif_path), str(out_dir / f"{stem}_viewer.html"),
                style=args.style, color=args.color)
    render_png(str(cif_path),  str(out_dir / f"{stem}_render.png"),
               dpi=args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
