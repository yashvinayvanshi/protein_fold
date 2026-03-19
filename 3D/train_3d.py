"""
train_3d.py — Train the 3-D protein structure predictor.

Usage
-----
    python 3D/train_3d.py [--cif_dir 3D/pdb_dataset] [--epochs 50] [--batch_size 8]

Outputs (all in 3D/outputs/)
----------------------------
    checkpoints/best_model_3d.pt    best checkpoint (lowest val loss)
    checkpoints/last_model_3d.pt    final-epoch checkpoint
    checkpoints/config_3d.json      model hyper-parameters
    logs/train_3d_<timestamp>.log   full training log
    train_results_3d.json           per-epoch metrics
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_loader_3d import get_dataloaders, VOCAB_SIZE, PAD_IDX
from model_3d import (
    DEFAULT_CONFIG, build_model, save_checkpoint,
)


# ── Kabsch RMSD ───────────────────────────────────────────────────────────────

def kabsch_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute RMSD between pred and true after optimal superposition (Kabsch).
    Both arrays have shape (L, 3).
    """
    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)
    H      = pred_c.T @ true_c
    U, _, Vt = np.linalg.svd(H)
    d      = np.linalg.det(Vt.T @ U.T)
    D      = np.diag([1.0, 1.0, d])
    R      = Vt.T @ D @ U.T
    pred_r = pred_c @ R.T
    diff   = pred_r - true_c
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_3d_{ts}.log"

    logger = logging.getLogger("train_3d")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


# ── Single training epoch ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, logger):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    bar = tqdm(loader, desc="  Train", leave=False, unit="batch",
               file=sys.stdout, dynamic_ncols=True)

    for batch in bar:
        encoded = batch["encoded"].to(device)
        coords  = batch["coords"].to(device)
        mask    = batch["mask"].to(device)

        optimizer.zero_grad()
        pred   = model(encoded, mask)            # (B, L, 3)

        # MSE loss on valid (non-padded) residues only
        valid  = mask.unsqueeze(-1).expand_as(pred)
        loss   = criterion(pred[valid], coords[valid])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")

    bar.close()
    return total_loss / max(n_batches, 1)


# ── Validation epoch ──────────────────────────────────────────────────────────

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    rmsds      = []

    with torch.no_grad():
        bar = tqdm(loader, desc="  Val  ", leave=False, unit="batch",
                   file=sys.stdout, dynamic_ncols=True)

        for batch in bar:
            encoded = batch["encoded"].to(device)
            coords  = batch["coords"].to(device)
            mask    = batch["mask"].to(device)

            pred  = model(encoded, mask)
            valid = mask.unsqueeze(-1).expand_as(pred)
            loss  = criterion(pred[valid], coords[valid])

            total_loss += loss.item()
            n_batches  += 1

            # Per-protein RMSD
            pred_np  = pred.cpu().numpy()
            true_np  = coords.cpu().numpy()
            mask_np  = mask.cpu().numpy()
            for b in range(pred_np.shape[0]):
                m = mask_np[b]
                if m.sum() < 4:
                    continue
                try:
                    rmsd = kabsch_rmsd(pred_np[b][m], true_np[b][m])
                    rmsds.append(rmsd)
                except Exception:
                    pass

        bar.close()

    mean_loss = total_loss / max(n_batches, 1)
    mean_rmsd = float(np.mean(rmsds)) if rmsds else float("nan")
    return mean_loss, mean_rmsd


# ── Main training function ────────────────────────────────────────────────────

def main():
    # ── argument parsing ──────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(description="Train 3-D protein structure predictor")
    ap.add_argument("--cif_dir",    default="3D/pdb_dataset",
                    help="Directory containing .cif files")
    ap.add_argument("--output_dir", default="3D/outputs",
                    help="Root output directory")
    ap.add_argument("--max_len",    type=int,   default=256)
    ap.add_argument("--max_files",  type=int,   default=None,
                    help="Limit number of CIF files (useful for quick runs)")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch_size", type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--val_frac",   type=float, default=0.10)
    ap.add_argument("--patience",   type=int,   default=8,
                    help="LR scheduler patience (epochs without val improvement)")
    ap.add_argument("--embed_dim",  type=int,   default=64)
    ap.add_argument("--num_filters",type=int,   default=192)
    ap.add_argument("--lstm_hidden",type=int,   default=256)
    ap.add_argument("--lstm_layers",type=int,   default=2)
    ap.add_argument("--dropout",    type=float, default=0.3)
    args = ap.parse_args()

    # ── resolve paths ─────────────────────────────────────────────────────────
    root       = Path(__file__).parent.parent
    cif_dir    = root / args.cif_dir
    out_dir    = root / args.output_dir
    ckpt_dir   = out_dir / "checkpoints"
    log_dir    = out_dir / "logs"
    for d in (ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)

    # ── device ────────────────────────────────────────────────────────────────
    device = (
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading CIF files from: {cif_dir}")
    train_loader, val_loader, _ = get_dataloaders(
        str(cif_dir),
        max_len    = args.max_len,
        batch_size = args.batch_size,
        val_frac   = args.val_frac,
        max_files  = args.max_files,
        logger     = logger,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )

    # ── model ─────────────────────────────────────────────────────────────────
    config = {
        "vocab_size":  VOCAB_SIZE,
        "embed_dim":   args.embed_dim,
        "num_filters": args.num_filters,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "dropout":     args.dropout,
        "pad_idx":     PAD_IDX,
        "max_len":     args.max_len,
    }
    model = build_model(config, device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Config: {config}")

    # save config
    cfg_path = ckpt_dir / "config_3d.json"
    cfg_path.write_text(json.dumps(config, indent=2))

    # ── optimizer / scheduler / loss ─────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience,
        factor=0.5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = []

    logger.info("=" * 60)
    logger.info(f"Starting training: {args.epochs} epochs")
    logger.info("=" * 60)

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, logger
        )
        val_loss, val_rmsd = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        ep_time = time.time() - ep_start
        logger.info(
            f"  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_RMSD={val_rmsd:.2f} Å  "
            f"lr={current_lr:.2e}  "
            f"time={ep_time:.1f}s"
        )

        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_rmsd":   round(val_rmsd,   4),
            "lr":         current_lr,
        }
        history.append(record)

        # Save last checkpoint
        save_checkpoint(model, config, str(ckpt_dir / "last_model_3d.pt"))

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config, str(ckpt_dir / "best_model_3d.pt"))
            logger.info(f"  ✓ New best checkpoint (val_loss={val_loss:.4f})")

    # ── final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    best_ep    = min(history, key=lambda r: r["val_loss"])

    logger.info("\n" + "=" * 60)
    logger.info("Training complete")
    logger.info(f"  Total time : {total_time/60:.1f} min")
    logger.info(
        f"  Best epoch : {best_ep['epoch']}  "
        f"val_loss={best_ep['val_loss']:.4f}  "
        f"val_RMSD={best_ep['val_rmsd']:.2f} Å"
    )
    logger.info("=" * 60)

    # Save results JSON
    results = {
        "config":       config,
        "best_epoch":   best_ep,
        "history":      history,
        "total_time_s": round(total_time, 1),
    }
    results_path = out_dir / "train_results_3d.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
