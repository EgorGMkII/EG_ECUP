"""Script 08: Fit Walk-Forward Classification & Amount Stacks."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
from scipy.special import expit, logit

from src.specialized_hurdle.diagnostics.classifier_metrics import compute_classifier_metrics
from src.specialized_hurdle.stacking.amount_ridge_stack import (
    fit_amount_ridge_stack,
    predict_amount_ridge_stack,
)
from src.specialized_hurdle.stacking.soft_classifier_stack import (
    fit_soft_classifier_stack,
    predict_soft_classifier_stack,
)


def main():
    print("=" * 80)
    print("08: FIT WALK-FORWARD CLASSIFICATION & AMOUNT STACKS")
    print("=" * 80)

    oof_dir = Path("artifacts/specialized_hurdle/oof")
    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_oof = pl.read_parquet(oof_dir / "full_walkforward_oof_matrix.parquet")
    print(f"[*] Loaded full walk-forward OOF matrix: {len(df_oof):,} rows.")

    # 1. Walk-Forward Meta-Validation
    meta_tests = [
        ("Meta_Test_1", ["2025-09-15", "2025-10-13", "2025-10-27"], "2025-11-10"),
        ("Meta_Test_2", ["2025-09-15", "2025-10-13", "2025-10-27", "2025-11-10"], "2025-11-24"),
        ("Meta_Test_3", ["2025-09-15", "2025-10-13", "2025-10-27", "2025-11-10", "2025-11-24"], "2025-12-08"),
        ("Meta_Test_4", ["2025-09-15", "2025-10-13", "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08"], "2025-12-15"),
    ]

    meta_results = []

    for name, train_anchors, test_anchor in meta_tests:
        df_train_meta = df_oof.filter(pl.col("anchor").is_in(train_anchors))
        df_test_meta = df_oof.filter(pl.col("anchor") == test_anchor)

        if len(df_train_meta) == 0 or len(df_test_meta) == 0:
            continue

        # Fit Soft React Stack
        train_inact = df_train_meta.filter(pl.col("was_active") == 0)
        test_inact = df_test_meta.filter(pl.col("was_active") == 0)

        # Convert logits to probs for stack fitting
        p_react_train = expit(train_inact["cb_react_logit"].to_numpy().reshape(-1, 1))
        p_react_test = expit(test_inact["cb_react_logit"].to_numpy().reshape(-1, 1))

        react_stack = fit_soft_classifier_stack(p_react_train, train_inact["will_buy"].to_numpy(), ["CatBoost"])
        p_react_pred = predict_soft_classifier_stack(react_stack, p_react_test)
        m_react = compute_classifier_metrics(test_inact["will_buy"].to_numpy(), p_react_pred)

        # Fit Soft Churn Stack
        train_act = df_train_meta.filter(pl.col("was_active") == 1)
        test_act = df_test_meta.filter(pl.col("was_active") == 1)

        p_churn_train = expit(train_act["cb_churn_logit"].to_numpy().reshape(-1, 1))
        p_churn_test = expit(test_act["cb_churn_logit"].to_numpy().reshape(-1, 1))

        churn_stack = fit_soft_classifier_stack(p_churn_train, (1 - train_act["will_buy"].to_numpy()), ["CatBoost"])
        p_churn_pred = predict_soft_classifier_stack(churn_stack, p_churn_test)
        m_churn = compute_classifier_metrics((1 - test_act["will_buy"].to_numpy()), p_churn_pred)

        print(f"[{name}] Test Anchor: {test_anchor} | React AUC: {m_react['roc_auc']:.4f} (LL: {m_react['log_loss']:.4f}) | Churn AUC: {m_churn['roc_auc']:.4f} (LL: {m_churn['log_loss']:.4f})")

        meta_results.append({
            "meta_test_id": name,
            "test_anchor": test_anchor,
            "n_train_anchors": len(train_anchors),
            "react_roc_auc": m_react["roc_auc"],
            "react_log_loss": m_react["log_loss"],
            "react_brier": m_react["brier_score"],
            "churn_roc_auc": m_churn["roc_auc"],
            "churn_log_loss": m_churn["log_loss"],
            "churn_brier": m_churn["brier_score"],
            "react_temp": react_stack.temperature,
            "churn_temp": churn_stack.temperature,
        })

    df_meta = pl.DataFrame(meta_results)
    out_csv = reports_dir / "walk_forward_meta_results.csv"
    df_meta.write_csv(out_csv)
    print(f"\n[+] Saved walk-forward meta results to {out_csv}")


if __name__ == "__main__":
    main()
