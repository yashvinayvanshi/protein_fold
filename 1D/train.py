"""
train.py — Training script for protein secondary structure prediction (Q3).

Usage:
    python train.py                          # defaults
    python train.py --epochs 50 --lr 5e-4
    python train.py --batch_size 128 --dropout 0.4

Outputs (written to --save_dir, default: checkpoints/):
    best_model.pt          — state dict of the epoch with highest val Q3 accuracy
    last_model.pt          — state dict after the final epoch
    config.json            — model + training hyperparameters (needed by predict.py)
    validation_results.json — full per-epoch metrics history

Logs (written to --log_dir, default: logs/):
    train_YYYYMMDD_HHMMSS.log — console mirror with timestamps

Pipeline position: data_loader + model → [this script] → saved model + results
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_loader import (
    NUM_CLASSES, PAD_IDX, PAD_LABEL, VOCAB_SIZE, IDX_TO_SS3,
    get_dataloaders,
)
from model import ProteinSSPredictor, build_model


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> str:
    """
    Configure root logger to write to both console and a timestamped log file.

    Input:
        log_dir (str) — directory where the log file will be created

    Output:
        log_path (str) — absolute path of the created log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log file: {log_path}")
    return log_path


# ── Training step ──────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run one full pass over the training DataLoader and update model weights.

    Input:
        model     — ProteinSSPredictor (set to train mode internally)
        loader    — DataLoader yielding (seq, labels, mask) batches
        optimizer — torch optimiser
        criterion — CrossEntropyLoss(ignore_index=PAD_LABEL)
        device    — torch.device

    Output:
        dict {
            "loss"   : float — mean cross-entropy loss over all batches,
            "q3_acc" : float — Q3 accuracy over real (non-padded) residues
        }
    """
    model.train()
    total_loss = 0.0
    correct = total = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for seq, lbl, mask in pbar:
        seq, lbl, mask = seq.to(device), lbl.to(device), mask.to(device)

        optimizer.zero_grad()
        logits = model(seq)                                          # (batch, seq_len, C)
        loss   = criterion(logits.view(-1, NUM_CLASSES), lbl.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()

        # Accuracy only over real residues (mask == True)
        preds = logits.argmax(dim=-1)              # (batch, seq_len)
        correct += (preds[mask] == lbl[mask]).sum().item()
        total   += mask.sum().item()

        running_acc = correct / total if total > 0 else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", q3=f"{running_acc:.4f}")

    return {
        "loss":   total_loss / len(loader),
        "q3_acc": correct / total if total > 0 else 0.0,
    }


# ── Validation step ────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on the validation DataLoader; compute Q3 accuracy and per-class metrics.

    Input:
        model     — ProteinSSPredictor (set to eval mode internally)
        loader    — DataLoader yielding (seq, labels, mask) batches
        criterion — CrossEntropyLoss(ignore_index=PAD_LABEL)
        device    — torch.device

    Output:
        dict {
            "loss"               : float,
            "q3_acc"             : float  — overall Q3 accuracy over real residues,
            "per_class_precision": dict[str→float]  — {C, H, E},
            "per_class_recall"   : dict[str→float]  — {C, H, E},
            "per_class_f1"       : dict[str→float]  — {C, H, E}
        }
    """
    model.eval()
    total_loss = 0.0

    # Counts for confusion matrix: shape (NUM_CLASSES,)
    class_correct   = np.zeros(NUM_CLASSES, dtype=np.int64)  # TP per class
    class_total     = np.zeros(NUM_CLASSES, dtype=np.int64)  # actual positives per class
    class_predicted = np.zeros(NUM_CLASSES, dtype=np.int64)  # predicted positives per class

    with torch.no_grad():
        pbar = tqdm(loader, desc="    val", leave=False, unit="batch",
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for seq, lbl, mask in pbar:
            seq, lbl, mask = seq.to(device), lbl.to(device), mask.to(device)

            logits = model(seq)
            loss   = criterion(logits.view(-1, NUM_CLASSES), lbl.view(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)                    # (batch, seq_len)
            real_preds = preds[mask].cpu().numpy()
            real_lbls  = lbl[mask].cpu().numpy()

            for c in range(NUM_CLASSES):
                class_correct[c]   += ((real_preds == c) & (real_lbls == c)).sum()
                class_total[c]     += (real_lbls == c).sum()
                class_predicted[c] += (real_preds == c).sum()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    overall_q3_acc = class_correct.sum() / class_total.sum() if class_total.sum() > 0 else 0.0

    # Per-class precision, recall, F1
    per_class_precision = {}
    per_class_recall    = {}
    per_class_f1        = {}
    for c in range(NUM_CLASSES):
        name = IDX_TO_SS3[c]
        prec = class_correct[c] / class_predicted[c] if class_predicted[c] > 0 else 0.0
        rec  = class_correct[c] / class_total[c]     if class_total[c]     > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec)          if (prec + rec)      > 0 else 0.0
        per_class_precision[name] = round(float(prec), 4)
        per_class_recall[name]    = round(float(rec),  4)
        per_class_f1[name]        = round(float(f1),   4)

    return {
        "loss":                total_loss / len(loader),
        "q3_acc":              float(overall_q3_acc),
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
        "per_class_f1":        per_class_f1,
    }


# ── Main training loop ─────────────────────────────────────────────────────────

def main(args):
    setup_logging(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    logging.info("=== Loading data ===")
    train_loader, val_loader, max_len = get_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_len=args.max_len,
        filter_nonstd=not args.keep_nonstd,
        seed=args.seed,
    )
    logging.info(f"Resolved max_len: {max_len}")

    # ── Model config (also persisted for inference) ───────────────────────────
    config = {
        "vocab_size":  VOCAB_SIZE,
        "embed_dim":   args.embed_dim,
        "num_filters": args.num_filters,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "num_classes": NUM_CLASSES,
        "dropout":     args.dropout,
        "pad_idx":     PAD_IDX,
        "max_len":     max_len,
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    logging.info("=== Building model ===")
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {n_params:,}")
    logging.info(f"Architecture:\n{model}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Reduce LR when val Q3 accuracy plateaus for `patience` epochs
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=args.patience, factor=0.5)

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # Persist config so predict.py can reconstruct the model without any CLI args
    config_path = os.path.join(args.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Config saved to {config_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logging.info(f"=== Training for {args.epochs} epochs ===")
    best_val_acc = 0.0
    history = []

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch",
                     bar_format="{l_bar}{bar:40}{r_bar}")

    for epoch in epoch_bar:
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step(val_m["q3_acc"])

        epoch_bar.set_postfix(
            tr_loss=f"{train_m['loss']:.4f}",
            tr_Q3=f"{train_m['q3_acc']:.4f}",
            val_loss=f"{val_m['loss']:.4f}",
            val_Q3=f"{val_m['q3_acc']:.4f}",
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Build epoch record for JSON history
        record = {
            "epoch":               epoch,
            "train_loss":          round(train_m["loss"],   4),
            "train_q3_acc":        round(train_m["q3_acc"], 4),
            "val_loss":            round(val_m["loss"],     4),
            "val_q3_acc":          round(val_m["q3_acc"],   4),
            "val_per_class_precision": val_m["per_class_precision"],
            "val_per_class_recall":    val_m["per_class_recall"],
            "val_per_class_f1":        val_m["per_class_f1"],
            "lr":                  current_lr,
        }
        history.append(record)

        # Console / file log
        logging.info(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_m['loss']:.4f}  train_Q3={train_m['q3_acc']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  val_Q3={val_m['q3_acc']:.4f}  "
            f"lr={current_lr:.2e}"
        )
        logging.info(
            f"  Val precision  C={val_m['per_class_precision']['C']:.4f}  "
            f"H={val_m['per_class_precision']['H']:.4f}  "
            f"E={val_m['per_class_precision']['E']:.4f}"
        )
        logging.info(
            f"  Val recall     C={val_m['per_class_recall']['C']:.4f}  "
            f"H={val_m['per_class_recall']['H']:.4f}  "
            f"E={val_m['per_class_recall']['E']:.4f}"
        )
        logging.info(
            f"  Val F1         C={val_m['per_class_f1']['C']:.4f}  "
            f"H={val_m['per_class_f1']['H']:.4f}  "
            f"E={val_m['per_class_f1']['E']:.4f}"
        )

        # Save best checkpoint
        if val_m["q3_acc"] > best_val_acc:
            best_val_acc = val_m["q3_acc"]
            best_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            logging.info(f"  ✓ New best model → val_Q3={best_val_acc:.4f} saved to {best_path}")

    # ── Post-training saves ───────────────────────────────────────────────────
    last_path = os.path.join(args.save_dir, "last_model.pt")
    torch.save(model.state_dict(), last_path)
    logging.info(f"Last model saved to {last_path}")

    results_path = os.path.join(args.save_dir, "validation_results.json")
    summary = {"best_val_q3_acc": best_val_acc, "history": history}
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Validation results saved to {results_path}")

    logging.info(f"=== Done — Best Val Q3 Accuracy: {best_val_acc:.4f} ===")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN+BiLSTM model to predict protein secondary structure (Q3)."
    )

    # Paths
    parser.add_argument("--csv_path",   default="2018-06-06-pdb-intersect-pisces.csv",
                        help="Path to the training CSV dataset")
    parser.add_argument("--save_dir",   default="checkpoints",
                        help="Directory to save model checkpoints and config")
    parser.add_argument("--log_dir",    default="logs",
                        help="Directory to write training log files")

    # Training
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--patience",   type=int,   default=5,
                        help="LR scheduler patience (epochs without val improvement)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--keep_nonstd", action="store_true",
                        help="If set, keep sequences with non-standard amino acids")

    # Sequence
    parser.add_argument("--max_len",    type=int,   default=512,
                        help="Max sequence length; sequences longer than this are truncated (default: 512)")

    # Model architecture
    parser.add_argument("--embed_dim",   type=int,   default=64)
    parser.add_argument("--num_filters", type=int,   default=64,
                        help="Filters per kernel size in the CNN block")
    parser.add_argument("--lstm_hidden", type=int,   default=256)
    parser.add_argument("--lstm_layers", type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)

    args = parser.parse_args()
    main(args)
