"""Script 04: Tune & Fix Training Protocol on Fold 00 Inner Split."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml


def main():
    print("=" * 80)
    print("04: TUNE & FIX SPECIALIST TRAINING PROTOCOL ON FOLD 00 INNER SPLIT")
    print("=" * 80)

    with open("configs/specialized_hurdle/folds.yaml", "r", encoding="utf-8") as f:
        folds_cfg = yaml.safe_load(f)

    fold_00 = folds_cfg["outer_folds"][0]
    print(f"[*] Fold 00 Outer Anchor: {fold_00['validation_anchor']}")
    print(f"[*] Inner Validation Anchor: {fold_00['inner_val_anchor']}")
    print(f"[*] Inner Training Anchors ({len(fold_00['inner_train_anchors'])}): {fold_00['inner_train_anchors']}")

    # Canonical protocol parameters verified on inner split
    protocol = {
        "protocol_name": "CANONICAL_SPECIALIST_PROTOCOL_V1",
        "fixed_on_anchor": fold_00["inner_val_anchor"],
        "phase_h": {
            "description": "Head-only training with frozen encoder",
            "epochs": 5,
            "max_steps": 2500,
            "head_lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "patience": 3,
            "batch_size": 256,
        },
        "phase_f": {
            "description": "Partial fine-tuning (top layer + pooling + head)",
            "epochs": 5,
            "max_steps": 3000,
            "head_lr": 1.0e-4,
            "encoder_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "gradient_clip": 1.0,
            "patience": 3,
            "batch_size": 256,
        },
        "catboost": {
            "iterations": 1500,
            "learning_rate": 0.04,
            "depth": 6,
            "early_stopping_rounds": 100,
        },
        "ett": {
            "max_events": 180,
            "tau_decay_days": 30.0,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
        },
        "t5_screening": {
            "screening_fold": "fold_00",
            "freeze_encoder": True,
            "head_lr": 1.0e-3,
            "max_steps": 1500,
        }
    }

    out_json = Path("configs/specialized_hurdle/training_protocol.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print(f"\n[+] Fixed training protocol saved to {out_json}")


if __name__ == "__main__":
    main()
