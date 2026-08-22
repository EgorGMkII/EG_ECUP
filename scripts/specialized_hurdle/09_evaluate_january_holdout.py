"""Script 09: Single Untouched Evaluation on January Holdout (2026-01-14)."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
from scipy.special import expit
import yaml

from src.specialized_hurdle.inference.external_hurdle import assemble_external_hurdle
from src.specialized_hurdle.specialists.train_catboost_specialist import (
    train_fold_catboost_specialists,
)


def main():
    print("=" * 80)
    print("09: TRAIN JANUARY FOLD-SAFE MODELS & EVALUATE UNTOUCHED JANUARY HOLDOUT")
    print("=" * 80)

    feature_store_dir = Path("artifacts/specialized_hurdle/feature_store")
    reports_dir = Path("artifacts/specialized_hurdle/reports")
    val_dir = Path("artifacts/specialized_hurdle/validation")
    jan_ckpt_dir = Path("artifacts/specialized_hurdle/specialist_checkpoints/january_holdout")
    reports_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    jan_ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/specialized_hurdle/folds.yaml", "r", encoding="utf-8") as f:
        folds_cfg = yaml.safe_load(f)

    jan_cfg = folds_cfg["final_january_holdout"]
    train_anchors = jan_cfg["train_anchors"]
    val_anchor = jan_cfg["validation_anchor"]

    print(f"[*] Training January Fold-Safe Specialists on all {len(train_anchors)} legal anchors...")
    cb_preds = train_fold_catboost_specialists(
        feature_store_dir=feature_store_dir,
        train_anchors=train_anchors,
        val_anchor=val_anchor,
        out_dir=jan_ckpt_dir,
    )

    react_logits = cb_preds["react_logits"]
    churn_logits = cb_preds["churn_logits"]
    amount_z = cb_preds["amount_z"]
    was_act = cb_preds["was_active"]
    will_buy = cb_preds["will_buy"]
    y_rub = cb_preds["y_rub"]
    z_true = np.log1p(np.maximum(0.0, y_rub))

    p_react = expit(react_logits)
    p_churn = expit(churn_logits)

    # 1. Clean External Hurdle Assembly (alpha = 1.0)
    p_buy, fact_z, pred_gmv = assemble_external_hurdle(
        p_react, p_churn, amount_z, was_act, alpha=1.0
    )

    rmsle = float(np.sqrt(np.mean((np.log1p(pred_gmv) - np.log1p(y_rub)) ** 2)))

    # State transitions
    st_0_0 = (was_act == 0) & (will_buy == 0)
    st_0_1 = (was_act == 0) & (will_buy == 1)
    st_1_0 = (was_act == 1) & (will_buy == 0)
    st_1_1 = (was_act == 1) & (will_buy == 1)

    mse_0_0 = float(np.mean((fact_z[st_0_0] - z_true[st_0_0]) ** 2))
    mse_0_1 = float(np.mean((fact_z[st_0_1] - z_true[st_0_1]) ** 2))
    mse_1_0 = float(np.mean((fact_z[st_1_0] - z_true[st_1_0]) ** 2))
    mse_1_1 = float(np.mean((fact_z[st_1_1] - z_true[st_1_1]) ** 2))

    print("\n" + "=" * 80)
    print("UNTOUCHED JANUARY HOLDOUT RESULTS (2026-01-14)")
    print("=" * 80)
    print(f"[*] January RMSLE = {rmsle:.5f}")
    print(f"    0->0 MSE: {mse_0_0:.4f}")
    print(f"    0->1 MSE: {mse_0_1:.4f}")
    print(f"    1->0 MSE: {mse_1_0:.4f}")
    print(f"    1->1 MSE: {mse_1_1:.4f}")

    jan_summary = [{
        "validation_anchor": "2026-01-14",
        "model_system": "EXTERNAL_SPECIALIZED_HURDLE",
        "rmsle": rmsle,
        "mse_0_0": mse_0_0,
        "mse_0_1": mse_0_1,
        "mse_1_0": mse_1_0,
        "mse_1_1": mse_1_1,
        "alpha": 1.0,
    }]
    pl.DataFrame(jan_summary).write_csv(reports_dir / "january_final_metrics.csv")
    print(f"\n[+] Saved January final metrics to {reports_dir / 'january_final_metrics.csv'}")


if __name__ == "__main__":
    main()
