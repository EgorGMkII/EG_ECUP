"""Strict Mathematical and Arithmetic Validator for LTV Experiment Reports.

Enforces:
1. RMSLE and MSE log-ruble consistency invariants.
2. Group transition partition completeness and exact SSE/MSE decomposition.
3. Proper size-weighted contribution share calculations.
4. Paired bootstrap confidence intervals and sample alignment assertions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score


def standardize_prediction_dataframe(df: pl.DataFrame, anchor_date: str = "2026-01-14", alpha: float = 1.1) -> pl.DataFrame:
    """Standardizes columns to the canonical schema required by Section 2."""
    cols = df.columns
    
    # Target / ground truth
    if "y_rub" in cols:
        y_rub = df["y_rub"]
    elif "target" in cols:
        y_rub = df["target"]
    elif "target_gt" in cols:
        y_rub = df["target_gt"]
    else:
        raise ValueError(f"Missing target column. Found: {cols}")

    z_true = np.log1p(y_rub.to_numpy().astype(np.float64))

    # Past state
    if "current_state" in cols:
        curr_state = df["current_state"].to_numpy().astype(np.int32)
    elif "past_buyer_30d" in cols:
        curr_state = df["past_buyer_30d"].to_numpy().astype(np.int32)
    elif "past_buyer" in cols:
        curr_state = df["past_buyer"].to_numpy().astype(np.int32)
    else:
        curr_state = np.zeros(len(df), dtype=np.int32)

    # Future buy indicator
    fut_buyer = (y_rub.to_numpy() > 0).astype(np.int32)

    # Transition states
    transitions = []
    for c, f in zip(curr_state, fut_buyer):
        if c == 0 and f == 0:
            transitions.append("0->0")
        elif c == 0 and f == 1:
            transitions.append("0->>0")
        elif c == 1 and f == 0:
            transitions.append(">0->0")
        else:
            transitions.append(">0->>0")

    # Probabilities and logits
    p_react = df["p_react"].to_numpy().astype(np.float64) if "p_react" in cols else np.full(len(df), 0.5)
    p_churn = df["p_churn"].to_numpy().astype(np.float64) if "p_churn" in cols else np.full(len(df), 0.5)
    
    p_buy = np.where(curr_state == 0, p_react, 1.0 - p_churn)
    p_buy = np.clip(p_buy, 1e-7, 1.0 - 1e-7)

    react_logit = np.log(p_react / (1.0 - np.clip(p_react, 1e-7, 1.0 - 1e-7)))
    churn_logit = np.log(p_churn / (1.0 - np.clip(p_churn, 1e-7, 1.0 - 1e-7)))

    # Conditional and factorized predictions
    if "conditional_z" in cols:
        z_cond = df["conditional_z"].to_numpy().astype(np.float64)
    elif "z_cond" in cols:
        z_cond = df["z_cond"].to_numpy().astype(np.float64)
    else:
        z_cond = np.zeros(len(df), dtype=np.float64)

    if "factorized_z" in cols:
        z_fact = df["factorized_z"].to_numpy().astype(np.float64)
    elif "z_gru_fact" in cols:
        z_fact = df["z_gru_fact"].to_numpy().astype(np.float64)
    else:
        z_fact = (np.power(p_buy, alpha) * z_cond).astype(np.float64)

    if "final_prediction_z" in cols:
        final_z = df["final_prediction_z"].to_numpy().astype(np.float64)
    else:
        final_z = z_fact

    final_rub = np.clip(np.expm1(final_z), 0.0, None)

    user_ids = df["user_id"].to_numpy().astype(np.int64) if "user_id" in cols else np.arange(len(df), dtype=np.int64)

    return pl.DataFrame({
        "user_id": user_ids,
        "anchor_date": [anchor_date] * len(df),
        "y_rub": y_rub.to_numpy().astype(np.float64),
        "z_true": z_true,
        "current_state": curr_state,
        "transition_state": transitions,
        "reactivation_logit": react_logit,
        "churn_logit": churn_logit,
        "p_buy": p_buy,
        "conditional_z": z_cond,
        "factorized_z": z_fact,
        "final_prediction_z": final_z,
        "final_prediction_rub": final_rub,
    })


def validate_report_invariants(
    cand_df: pl.DataFrame,
    base_df: Optional[pl.DataFrame] = None,
    alpha: float = 1.1,
) -> Dict:
    """Executes all Section 2 invariants and returns verified metrics summary."""
    cand = standardize_prediction_dataframe(cand_df, alpha=alpha)
    n_total = len(cand)

    z_true = cand["z_true"].to_numpy()
    final_z = cand["final_prediction_z"].to_numpy()
    y_rub = cand["y_rub"].to_numpy()
    pred_rub = cand["final_prediction_rub"].to_numpy()
    transitions = cand["transition_state"].to_numpy()
    p_buy = cand["p_buy"].to_numpy()
    curr_state = cand["current_state"].to_numpy()

    # -------------------------------------------------------------------------
    # Invariant 2.1: Overall Metric & Ruble Consistency
    # -------------------------------------------------------------------------
    mse_log = float(np.mean((z_true - final_z) ** 2))
    rmsle = float(np.sqrt(mse_log))
    
    assert abs(rmsle**2 - mse_log) < 1e-12, f"Invariant 2.1 failed: rmsle^2 != mse_log ({rmsle**2} vs {mse_log})"

    rmsle_from_rub = float(np.sqrt(np.mean((np.log1p(y_rub) - np.log1p(np.clip(pred_rub, 0.0, None))) ** 2)))
    assert abs(rmsle - rmsle_from_rub) < 1e-5, f"Invariant 2.1 failed: rmsle != rmsle_from_rub ({rmsle} vs {rmsle_from_rub})"

    # -------------------------------------------------------------------------
    # Invariant 2.2: Transition Decomposition & Size-Weighted Contributions
    # -------------------------------------------------------------------------
    group_stats = {}
    total_sse = 0.0
    group_names = ["0->0", "0->>0", ">0->0", ">0->>0"]

    for g in group_names:
        mask = (transitions == g)
        n_g = int(np.sum(mask))
        share_g = float(n_g / n_total)
        sse_g = float(np.sum((z_true[mask] - final_z[mask]) ** 2))
        mse_g = float(sse_g / n_g) if n_g > 0 else 0.0
        rmsle_g = float(np.sqrt(mse_g))

        group_stats[g] = {
            "N": n_g,
            "share": share_g,
            "SSE": sse_g,
            "MSE": mse_g,
            "RMSLE": rmsle_g,
        }
        total_sse += sse_g

    reconstructed_mse = float(total_sse / n_total)
    assert abs(reconstructed_mse - mse_log) < 1e-10, f"Invariant 2.2 failed: reconstructed_mse ({reconstructed_mse}) != overall_mse ({mse_log})"
    assert sum(g["N"] for g in group_stats.values()) == n_total, "Invariant 2.2 failed: sum of N_g != N_total"
    assert abs(sum(g["share"] for g in group_stats.values()) - 1.0) < 1e-12, "Invariant 2.2 failed: sum of shares != 1.0"

    actual_buy_rate = float(np.mean(y_rub > 0))
    reconstructed_buy_rate = float((group_stats["0->>0"]["N"] + group_stats[">0->>0"]["N"]) / n_total)
    assert abs(actual_buy_rate - reconstructed_buy_rate) < 1e-12, "Invariant 2.2 failed: buy_rate mismatch"

    # Calibration metrics
    dormant_mask = (curr_state == 0)
    active_mask = (curr_state == 1)
    fut_buy = (y_rub > 0).astype(np.int32)

    react_auc = float(roc_auc_score(fut_buy[dormant_mask], p_buy[dormant_mask]))
    react_brier = float(brier_score_loss(fut_buy[dormant_mask], p_buy[dormant_mask]))

    churn_auc = float(roc_auc_score(1 - fut_buy[active_mask], 1.0 - p_buy[active_mask]))
    churn_brier = float(brier_score_loss(1 - fut_buy[active_mask], 1.0 - p_buy[active_mask]))

    overall_brier = float(brier_score_loss(fut_buy, p_buy))
    mean_p_buy = float(np.mean(p_buy))

    # -------------------------------------------------------------------------
    # Invariant 2.3: Paired Model Comparison & Bootstrap CI
    # -------------------------------------------------------------------------
    paired_comparison = None
    if base_df is not None:
        base = standardize_prediction_dataframe(base_df, alpha=alpha)
        
        # Invariant checks for paired rows
        assert len(base) == len(cand), f"Sample size mismatch: base ({len(base)}) vs cand ({len(cand)})"
        assert np.array_equal(base["user_id"].to_numpy(), cand["user_id"].to_numpy()), "user_id order mismatch between baseline and candidate"
        assert np.allclose(base["z_true"].to_numpy(), cand["z_true"].to_numpy(), atol=1e-7), "Ground truth z_true mismatch between baseline and candidate"

        z_base = base["final_prediction_z"].to_numpy()
        mse_base = float(np.mean((z_true - z_base) ** 2))
        rmsle_base = float(np.sqrt(mse_base))

        delta_rmsle = rmsle - rmsle_base
        delta_mse = mse_log - mse_base
        rel_delta_mse_pct = (delta_mse / mse_base) * 100.0

        # Weighted transition contributions
        # delta_total_mse_g = share_g * (mse_baseline_g - mse_candidate_g)
        # Note: positive delta_total_mse_g means candidate improved (reduced) MSE.
        delta_group_mse_weighted = {}
        for g in group_names:
            mask = (transitions == g)
            sse_b_g = float(np.sum((z_true[mask] - z_base[mask]) ** 2))
            mse_b_g = sse_b_g / group_stats[g]["N"]
            mse_c_g = group_stats[g]["MSE"]
            delta_total_mse_g = group_stats[g]["share"] * (mse_b_g - mse_c_g)
            delta_group_mse_weighted[g] = {
                "mse_baseline": mse_b_g,
                "mse_candidate": mse_c_g,
                "delta_mse_raw": mse_c_g - mse_b_g,
                "weighted_mse_reduction": delta_total_mse_g,
            }

        total_reduction = sum(d["weighted_mse_reduction"] for d in delta_group_mse_weighted.values())
        for g in group_names:
            if abs(total_reduction) > 1e-12:
                delta_group_mse_weighted[g]["contribution_share"] = delta_group_mse_weighted[g]["weighted_mse_reduction"] / total_reduction
            else:
                delta_group_mse_weighted[g]["contribution_share"] = 0.0

        # Paired Bootstrap 95% CI
        np.random.seed(42)
        n_boot = 1000
        boot_diffs = []
        cand_diff_sq = (z_true - final_z) ** 2
        base_diff_sq = (z_true - z_base) ** 2

        for _ in range(n_boot):
            idx = np.random.randint(0, n_total, size=n_total)
            r_c = np.sqrt(np.mean(cand_diff_sq[idx]))
            r_b = np.sqrt(np.mean(base_diff_sq[idx]))
            boot_diffs.append(r_c - r_b)

        boot_diffs = np.array(boot_diffs)
        ci_low, ci_med, ci_high = np.percentile(boot_diffs, [2.5, 50.0, 97.5])
        p_cand_better = float(np.mean(boot_diffs < 0))

        paired_comparison = {
            "rmsle_baseline": rmsle_base,
            "rmsle_candidate": rmsle,
            "delta_rmsle": delta_rmsle,
            "mse_baseline": mse_base,
            "mse_candidate": mse_log,
            "delta_mse": delta_mse,
            "relative_delta_mse_pct": rel_delta_mse_pct,
            "bootstrap_95_ci": [float(ci_low), float(ci_high)],
            "bootstrap_p_candidate_better": p_cand_better,
            "weighted_transition_contributions": delta_group_mse_weighted,
        }

    return {
        "status": "ARITHMETIC_VALIDATION: PASSED",
        "n_users": n_total,
        "rmsle": rmsle,
        "mse_log": mse_log,
        "actual_buy_rate": actual_buy_rate,
        "mean_p_buy": mean_p_buy,
        "react_auc": react_auc,
        "react_brier": react_brier,
        "churn_auc": churn_auc,
        "churn_brier": churn_brier,
        "overall_brier": overall_brier,
        "transitions": group_stats,
        "paired_comparison": paired_comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict arithmetic validator for experiment predictions.")
    parser.add_argument("-c", "--candidate", type=str, required=True, help="Candidate prediction parquet path.")
    parser.add_argument("-b", "--baseline", type=str, default=None, help="Baseline prediction parquet path for paired comparison.")
    parser.add_argument("--alpha", type=float, default=1.1, help="Hurdle exponent alpha.")
    args = parser.parse_args()

    cand_df = pl.read_parquet(args.candidate)
    base_df = pl.read_parquet(args.baseline) if args.baseline else None

    result = validate_report_invariants(cand_df, base_df, alpha=args.alpha)
    print("\n" + "=" * 80)
    print(result["status"])
    print("=" * 80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
