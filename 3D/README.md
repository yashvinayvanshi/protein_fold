# 3D Protein Structure Prediction

A deep learning pipeline that takes a chain of amino acids and predicts the
3-D structure it will fold into, outputs a `.cif` file, renders an interactive
HTML viewer and a static PNG, and validates predictions against experimental
structures via Kabsch-RMSD.

---

## Architecture

```
Amino-acid sequence
        │
        ▼
  [ Embedding ]  ← vocab size 22 (20 AAs + X + PAD)
        │
        ▼
[ Multi-scale CNN ]  ← parallel 1-D convolutions (k=3, 5, 7)
        │
        ▼
  [ BiLSTM × 2 ]  ← bidirectional, 256 hidden units per direction
        │
        ▼
   [ MLP head ]  ← Linear → ReLU → Dropout → Linear(3)
        │
        ▼
  (x, y, z) per residue  →  Cα backbone coordinates (centroid-centred)
        │
        ▼
  Written as mmCIF file  →  Rendered as HTML + PNG
```

**Model:** `ProteinFoldPredictor` — ~3.5 M trainable parameters
**Loss:** MSE on valid (non-padded) Cα coordinates
**Validation metric:** Kabsch-RMSD (Å) after optimal superposition

---

## Directory Structure

```
3D/
├── README.md                   ← you are here
├── requirements_3d.txt         ← all Python dependencies
│
├── fetch_data.py               ← downloads CIF files from RCSB PDB API
├── data_loader_3d.py           ← CIF parser, PyTorch Dataset/DataLoader
├── model_3d.py                 ← model definition + checkpoint helpers
├── train_3d.py                 ← training loop, logging, checkpointing
├── predict_3d.py               ← inference: sequence → .cif file
├── visualize_3d.py             ← matplotlib 3-D backbone plots
├── render_cif.py               ← py3Dmol HTML viewer + enhanced PNG renderer
│
├── pdb_dataset/                ← ~1000 experimental .cif files (training data)
│   └── *.cif
│
└── outputs/
    ├── checkpoints/
    │   ├── best_model_3d.pt    ← best checkpoint (lowest val loss)
    │   ├── last_model_3d.pt    ← final-epoch checkpoint
    │   └── config_3d.json      ← model hyper-parameters
    ├── logs/
    │   └── train_3d_<timestamp>.log
    ├── train_results_3d.json   ← per-epoch metrics (loss, RMSD, lr)
    └── predictions/
        ├── <name>.cif          ← predicted Cα backbone structure
        ├── <name>_report.json  ← prediction metadata + validation metrics
        ├── <name>_viewer.html  ← interactive 3-D viewer (py3Dmol)
        └── <name>_render.png   ← static high-quality PNG
```

---

## What each file does

| File | Description |
|---|---|
| `fetch_data.py` | Queries the RCSB PDB REST API for high-quality structures (resolution ≤ 2.0 Å, length 50–500 AA) and downloads up to 1000 `.cif` files into `pdb_dataset/` |
| `data_loader_3d.py` | Parses mmCIF files, extracts the one-letter sequence and Cα coordinates of the first protein chain, centres coordinates at the centroid, encodes sequences as integer tensors, and returns PyTorch `DataLoader` objects |
| `model_3d.py` | Defines `ProteinFoldPredictor` (Embedding → MultiScaleCNN → BiLSTM → MLP), factory functions `build_model` / `save_checkpoint` / `load_checkpoint` |
| `train_3d.py` | Full training loop with tqdm progress bars, per-epoch console + file logging, ReduceLROnPlateau scheduler, gradient clipping, best-model checkpointing, and JSON results export |
| `predict_3d.py` | Loads a checkpoint, runs inference on a sequence string or FASTA file, writes a valid mmCIF file (with `_cell` and `_symmetry` blocks), optionally validates against a reference CIF via Kabsch-RMSD, and saves a JSON report |
| `visualize_3d.py` | Lightweight matplotlib visualiser: single-structure 3-D backbone, predicted-vs-reference overlay, and 2-D projection panels |
| `render_cif.py` | Production renderer: **py3Dmol** (3Dmol.js) self-contained interactive HTML file + enhanced **matplotlib** dark-theme tube plot PNG with depth-shading and curvature-scaled spheres |

---

## Setup

### 1. Install dependencies

```bash
pip install -r 3D/requirements_3d.txt
```

> If your system Python is managed (e.g. Homebrew on macOS), use a specific
> version that already has PyTorch:
>
> ```bash
> python3.12 -m pip install -r 3D/requirements_3d.txt
> ```

Dependencies installed:

```
torch>=2.0.0       # model training & inference
numpy>=1.24.0      # numerical operations
tqdm>=4.65.0       # progress bars
scipy>=1.11.0      # SVD for Kabsch alignment
matplotlib>=3.7.0  # static PNG rendering
py3Dmol>=2.0.0     # interactive HTML molecular viewer
```

---

## Step-by-step execution

All commands are run from the **project root** (`protein_fold/`).

---

### Step 1 — (Optional) Download more CIF files

The `pdb_dataset/` folder already contains ~1000 files. To download a fresh
set:

```bash
python3.12 3D/fetch_data.py
```

---

### Step 2 — Train the model

```bash
python3.12 3D/train_3d.py
```

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--cif_dir` | `3D/pdb_dataset` | Folder with input `.cif` files |
| `--output_dir` | `3D/outputs` | Root folder for all outputs |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--max_files` | *(all)* | Limit CIF files loaded (useful for quick tests) |
| `--max_len` | `256` | Maximum sequence length (residues) |
| `--lr` | `1e-3` | Initial learning rate |
| `--patience` | `8` | LR scheduler patience (epochs) |

**Quick test run (3 epochs, 200 files):**

```bash
python3.12 3D/train_3d.py --epochs 3 --batch_size 8 --max_files 200
```

**Full run:**

```bash
python3.12 3D/train_3d.py --epochs 50 --batch_size 16
```

**What you see on the console:**

```
2026-03-19 15:07:06  INFO      Device: cpu
2026-03-19 15:07:06  INFO      Loading CIF files from: …/pdb_dataset
2026-03-19 15:07:07  INFO      Dataset: 200 proteins loaded, 0 skipped.
2026-03-19 15:07:07  INFO      Train batches: 23 | Val batches: 3

Epoch 1/3
  Train [batch progress bar]
  Val   [batch progress bar]
  train_loss=…  val_loss=…  val_RMSD=… Å  lr=…  time=…s
  ✓ New best checkpoint (val_loss=…)
```

**Outputs written:**

```
3D/outputs/checkpoints/best_model_3d.pt
3D/outputs/checkpoints/last_model_3d.pt
3D/outputs/checkpoints/config_3d.json
3D/outputs/logs/train_3d_<timestamp>.log
3D/outputs/train_results_3d.json
```

---

### Step 3 — Predict a structure

Provide an amino-acid sequence (one-letter codes):

```bash
python3.12 3D/predict_3d.py \
  --sequence "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAI"
```

From a FASTA file:

```bash
python3.12 3D/predict_3d.py --fasta my_protein.fasta
```

Give the output a custom name:

```bash
python3.12 3D/predict_3d.py --sequence "ACDE..." --name my_protein
```

**Outputs written:**

```
3D/outputs/predictions/predicted.cif
3D/outputs/predictions/predicted_report.json
```

---

### Step 4 — Validate against a reference CIF

Appending `--reference` computes the Kabsch-RMSD between the prediction and
an experimental structure:

```bash
python3.12 3D/predict_3d.py \
  --sequence "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAI" \
  --reference 3D/pdb_dataset/102L.cif
```

**Console output:**

```
Kabsch RMSD = 12.34 Å  (aligned 50 residues)
```

**Validation metrics in report JSON:**

```json
"validation": {
  "reference_pdb":    "102L",
  "predicted_length": 50,
  "reference_length": 185,
  "aligned_length":   50,
  "kabsch_rmsd_A":    12.34
}
```

> A lower RMSD (Å) means the predicted fold is closer to the experimental
> structure. State-of-the-art models (AlphaFold2) achieve < 2 Å on many
> proteins; this lightweight model is a learning baseline.

---

### Step 5 — Render the predicted CIF

```bash
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif
```

**Outputs:**

| File | Description |
|---|---|
| `predicted_viewer.html` | Interactive 3-D viewer — open in any browser. Drag to rotate, scroll to zoom, right-drag to translate |
| `predicted_render.png` | Dark-theme tube-plot with N→C plasma colouring, curvature-scaled Cα spheres, and depth shadow |

**Style options (affects HTML viewer):**

```bash
# Cartoon ribbon (default)
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif --style cartoon

# Ball-and-stick
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif --style stick

# Spheres
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif --style sphere

# Colour by secondary structure (Jmol scheme)
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif --color ssJmol

# Custom output directory
python3.12 3D/render_cif.py --cif 3D/outputs/predictions/predicted.cif \
                             --outdir 3D/outputs/predictions
```

**Render a reference CIF the same way:**

```bash
python3.12 3D/render_cif.py --cif 3D/pdb_dataset/102L.cif \
                             --outdir 3D/outputs/predictions
```

---

### Step 6 — (Optional) Lightweight matplotlib visualiser

For quick backbone plots without the full renderer:

```bash
# Single structure
python3.12 3D/visualize_3d.py --cif 3D/outputs/predictions/predicted.cif

# Overlay predicted vs reference (with RMSD annotation)
python3.12 3D/visualize_3d.py \
  --cif       3D/outputs/predictions/predicted.cif \
  --reference 3D/pdb_dataset/102L.cif

# Also save 2-D projection panels (XY / XZ / YZ)
python3.12 3D/visualize_3d.py \
  --cif 3D/outputs/predictions/predicted.cif --projections

# Predict from sequence and visualise immediately (no separate predict step)
python3.12 3D/visualize_3d.py --sequence "ACDEFGHIKLMNPQRSTVWY..."
```

---

## Full pipeline — one-liner sequence

```bash
# 1. Train
python3.12 3D/train_3d.py --epochs 50 --batch_size 16

# 2. Predict
python3.12 3D/predict_3d.py \
  --sequence "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAI" \
  --reference 3D/pdb_dataset/102L.cif \
  --name my_protein

# 3. Render
python3.12 3D/render_cif.py \
  --cif 3D/outputs/predictions/my_protein.cif \
  --style cartoon --color spectrum
```

---

## Output file reference

| Path | Created by | Contents |
|---|---|---|
| `outputs/checkpoints/best_model_3d.pt` | `train_3d.py` | Best model weights + config dict |
| `outputs/checkpoints/last_model_3d.pt` | `train_3d.py` | Final-epoch weights + config dict |
| `outputs/checkpoints/config_3d.json` | `train_3d.py` | Hyper-parameters (vocab size, hidden dims, …) |
| `outputs/logs/train_3d_<ts>.log` | `train_3d.py` | Full timestamped training log |
| `outputs/train_results_3d.json` | `train_3d.py` | Per-epoch train loss, val loss, val RMSD, lr |
| `outputs/predictions/<name>.cif` | `predict_3d.py` | Predicted Cα backbone in mmCIF format |
| `outputs/predictions/<name>_report.json` | `predict_3d.py` | Sequence, length, timing, validation RMSD |
| `outputs/predictions/<name>_viewer.html` | `render_cif.py` | Self-contained interactive py3Dmol viewer |
| `outputs/predictions/<name>_render.png` | `render_cif.py` | Dark-theme tube-plot PNG |
| `outputs/predictions/<name>_3d.png` | `visualize_3d.py` | Standard matplotlib 3-D backbone PNG |

---

## Notes

- **Coordinate frame:** The model predicts centroid-centred Cα coordinates.
  Absolute position and global orientation are arbitrary — RMSD is always
  computed after Kabsch superposition.
- **Sequence length:** Sequences longer than `--max_len` (default 256) are
  silently truncated. Adjust with `--max_len 512` if needed (increases memory).
- **CIF compatibility:** Predicted CIFs include `_cell` and `_symmetry` blocks
  set to P1 space group so they open without warnings in PyMOL, UCSF ChimeraX,
  CCP4, and Coot.
- **Training data:** All 1000 CIF files in `pdb_dataset/` are used by default.
  Structures with < 20 parseable Cα atoms are skipped automatically.
