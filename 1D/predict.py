"""
predict.py — Inference invoker for the trained protein secondary structure predictor.

Usage:
    # Single sequence via CLI
    python predict.py --sequence "ACDEFGHIKLMNPQRSTVWY"

    # Custom checkpoint directory
    python predict.py --sequence "NPVVHFFKNIVTPRTPPPSQ" --checkpoint_dir checkpoints

    # Use last model instead of best
    python predict.py --sequence "NPVVHFFKNIVTPRTPPPSQ" --use_last

    # As a Python module
    from predict import load_model, predict_sequence
    model, config = load_model("checkpoints", device)
    result = predict_sequence(model, "ACDEFGHIKL", config["max_len"], device)
    print(result["predicted_sst3"])

Output legend:
    C = Coil / Loop / Irregular
    H = α-Helix
    E = β-Strand

Pipeline position: saved checkpoints → [this module] → predicted SS3 string
"""

import argparse
import json

import numpy as np
import torch

from data_loader import AA_TO_IDX, IDX_TO_SS3, PAD_IDX, VOCAB_SIZE, encode_sequence
from model import build_model


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device: torch.device, use_last: bool = False) -> tuple:
    """
    Load a trained ProteinSSPredictor from a checkpoint directory.

    Input:
        checkpoint_dir (str)         — directory produced by train.py; must contain
                                        config.json and best_model.pt (or last_model.pt)
        device         (torch.device) — target device
        use_last       (bool)         — if True, load last_model.pt instead of best_model.pt

    Output:
        model  (ProteinSSPredictor) — model in eval mode on `device`
        config (dict)               — hyperparameters including max_len
    """
    config_path = f"{checkpoint_dir}/config.json"
    weights_file = "last_model.pt" if use_last else "best_model.pt"
    model_path  = f"{checkpoint_dir}/{weights_file}"

    with open(config_path) as f:
        config = json.load(f)

    model = build_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config


# ── Per-sequence inference ─────────────────────────────────────────────────────

def predict_sequence(
    model,
    sequence: str,
    max_len: int,
    device: torch.device,
) -> dict:
    """
    Predict Q3 secondary structure labels for a single amino acid sequence.

    Input:
        model    (ProteinSSPredictor) — loaded model in eval mode
        sequence (str)               — raw amino acid sequence (e.g. "ACDEFGHIKL...")
        max_len  (int)               — model's max sequence length (from config["max_len"])
        device   (torch.device)

    Output:
        dict {
            "sequence"       : str — input sequence, clipped to max_len,
            "predicted_sst3" : str — Q3 prediction per residue (same length as sequence),
            "length"         : int — number of residues predicted
        }
    """
    seq_clipped = sequence[:max_len]
    real_len    = len(seq_clipped)

    # Encode into a padded fixed-length array
    encoded = encode_sequence(seq_clipped, max_len)          # (max_len,) — padded to max_len
    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)  # (1, max_len)

    with torch.no_grad():
        logits = model(x)                                    # (1, max_len, num_classes)
        preds  = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (max_len,)

    # Decode only the real (non-padded) residue positions
    ss3_str = "".join(IDX_TO_SS3[int(p)] for p in preds[:real_len])

    return {
        "sequence":       seq_clipped,
        "predicted_sst3": ss3_str,
        "length":         real_len,
    }


def predict_batch(
    model,
    sequences: list,
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list:
    """
    Predict Q3 secondary structures for a list of sequences efficiently in batches.

    Input:
        model      (ProteinSSPredictor) — loaded model in eval mode
        sequences  (list[str])          — list of amino acid sequences
        max_len    (int)                — model's max sequence length
        device     (torch.device)
        batch_size (int)                — number of sequences per forward pass

    Output:
        list[dict] — one result dict per sequence, same format as predict_sequence()
    """
    model.eval()
    results = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs  = sequences[start : start + batch_size]
        real_lens   = [min(len(s), max_len) for s in batch_seqs]

        # Build integer array (batch, max_len)
        encoded_batch = np.stack([encode_sequence(s, max_len) for s in batch_seqs])
        x = torch.tensor(encoded_batch, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)                                 # (batch, max_len, C)
            preds  = logits.argmax(dim=-1).cpu().numpy()      # (batch, max_len)

        for i, (seq, rlen) in enumerate(zip(batch_seqs, real_lens)):
            ss3_str = "".join(IDX_TO_SS3[int(p)] for p in preds[i, :rlen])
            results.append({
                "sequence":       seq[:max_len],
                "predicted_sst3": ss3_str,
                "length":         rlen,
            })

    return results


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    """
    Print a prediction result in a human-readable aligned format.

    Input:
        result (dict) — output of predict_sequence() or an element of predict_batch()

    Output:
        None (prints to stdout)
    """
    seq  = result["sequence"]
    ss3  = result["predicted_sst3"]
    n    = result["length"]

    print(f"\n{'─' * 60}")
    print(f"Length : {n} residues")
    print(f"AA     : {seq}")
    print(f"SS3    : {ss3}")
    print(f"{'─' * 60}")
    print("Legend : C = Coil/Loop  |  H = α-Helix  |  E = β-Strand\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict protein secondary structure (Q3) from an amino acid sequence."
    )
    parser.add_argument(
        "--sequence", type=str, required=True,
        help="Amino acid sequence (e.g. ACDEFGHIKLMNPQRSTVWY)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory containing best_model.pt (or last_model.pt) and config.json"
    )
    parser.add_argument(
        "--use_last", action="store_true",
        help="Load last_model.pt instead of the default best_model.pt"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint_dir}  (device: {device})")
    model, config = load_model(args.checkpoint_dir, device, use_last=args.use_last)
    max_len = config["max_len"]
    print(f"Model ready — max_len={max_len}")

    result = predict_sequence(model, args.sequence.upper(), max_len, device)
    print_result(result)


if __name__ == "__main__":
    main()
