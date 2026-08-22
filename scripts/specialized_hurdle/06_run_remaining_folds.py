"""Script 06: Run & Audit Remaining Folds (Fold 01 .. Fold 06)."""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
import yaml

from src.specialized_hurdle.specialists.train_catboost_specialist import (
    train_fold_catboost_specialists,
)


def main():
    print("=" * 80)
    print("06: EXECUTE REMAINING FOLDS (FOLD 01 .. FOLD 06)")
    print("=" * 80)

    feature_store_dir = Path("artifacts/specialized_hurdle/feature_store")
    oof_dir = Path("artifacts/specialized_hurdle/oof")
    base_ckpt_dir = Path("artifacts/specialized_hurdle/specialist_checkpoints")
    oof_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/specialized_hurdle/folds.yaml", "r", encoding="utf-8") as f:
        folds_cfg = yaml.safe_load(f)

    users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()

    # Iterate over remaining outer folds 01 to 06
    for fold in folds_cfg["outer_folds"][1:]:
        f_id = fold["fold_id"]
        train_anchors = fold["train_anchors"]
        val_anchor = fold["validation_anchor"]
        f_ckpt_dir = base_ckpt_dir / f_id
        f_ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[*] Executing {f_id} (Validation Anchor: {val_anchor}) | {len(train_anchors)} Train Anchors...")
        t0 = time.time()

        cb_preds = train_fold_catboost_specialists(
            feature_store_dir=feature_store_dir,
            train_anchors=train_anchors,
            val_anchor=val_anchor,
            out_dir=f_ckpt_dir,
        )

        df_fold_oof = pl.DataFrame({
            "user_id": users_100k,
            "anchor": [val_anchor] * len(users_100k),
            "was_active": cb_preds["was_active"],
            "will_buy": cb_preds["will_buy"],
            "y_rub": cb_preds["y_rub"],
            "cb_react_logit": cb_preds["react_logits"],
            "cb_churn_logit": cb_preds["churn_logits"],
            "cb_amount_z": cb_preds["amount_z"],
        })
        oof_path = oof_dir / f"{f_id}.parquet"
        df_fold_oof.write_parquet(oof_path)

        dt = (time.time() - t0) / 60.0
        print(f"[+] {f_id} completed and saved to {oof_path.name} in {dt:.2f} min.")


if __name__ == "__main__":
    main()
