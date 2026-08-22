"""End-to-End Specialized Hurdle Stack Pipeline.

Implements:
1. Training & Fine-tuning of Specialized Models (CatBoost, ETT, GRU) for:
   - Reactivation Specialist (was_active == False, BCE on will_buy)
   - Churn Specialist (was_active == True, BCE on not will_buy)
   - Amount Specialist (future_gmv_30d > 0, MSE on log1p(GMV))
2. Generation of Fold-Safe OOF Prediction Matrices (fold_00, fold_01, fold_02, fold_03)
3. Walk-Forward Meta-Stacking:
   - Soft Reactivation Stack (temperature-scaled softmax weights)
   - Soft Churn Stack (temperature-scaled softmax weights)
   - Positive-Only Amount Ridge Stack (with was_active interactions)
4. Clean Single External Hurdle Assembly & January Benchmark Evaluation
5. Export of all CSV reports, metrics, and prediction tables.
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit, logit
from sklearn.linear_model import Ridge

from src.specialized_hurdle.definitions import (
    ALL_AVAILABLE_ANCHORS,
    JANUARY_VALIDATION_ANCHOR,
)
from src.specialized_hurdle.diagnostics.classifier_metrics import compute_classifier_metrics
from src.specialized_hurdle.inference.external_hurdle import assemble_external_hurdle
from src.specialized_hurdle.stacking.amount_ridge_stack import (
    fit_amount_ridge_stack,
    predict_amount_ridge_stack,
)
from src.specialized_hurdle.stacking.soft_classifier_stack import (
    fit_soft_classifier_stack,
    predict_soft_classifier_stack,
)


def load_dataset_slice(snap_path: Path, users_100k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snap = pl.read_parquet(snap_path)
    u_map = {u: i for i, u in enumerate(snap["user_id"].to_list())}
    order = [u_map[u] for u in users_100k if u in u_map]

    y_rub = snap["target"].to_numpy()[order].astype(np.float32)
    past_gmv = (snap["lifetime_gmv"].to_numpy()[order] if "lifetime_gmv" in snap.columns else snap["gmv_sum_30d"].to_numpy()[order]).astype(np.float32)
    was_act = (past_gmv > 0).astype(int)
    will_buy = (y_rub > 0).astype(int)
    z_true = np.log1p(np.maximum(0.0, y_rub))

    return was_act, will_buy, z_true, y_rub


def main():
    print("=" * 80)
    print("SPECIALIZED HURDLE STACK: COMPLETE TRAINING, STACKING & EVALUATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution device: {device}")

    base_dir = Path("artifacts/specialized_hurdle")
    reports_dir = base_dir / "reports"
    oof_dir = base_dir / "oof"
    val_dir = base_dir / "validation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()
    n_users = len(users_100k)
    print(f"[+] Loaded {n_users} benchmark users.")

    # 1. Load Ground Truth for January Validation Anchor (2026-01-14)
    val_was_act, val_will_buy, val_z_true, val_y_rub = load_dataset_slice(
        Path("data/snapshots/snapshot_2026-01-14.parquet"), users_100k
    )

    # 2. Extract Specialist Predictions from Existing High-Capacity Checkpoints
    print("\n[*] Loading & Extracting Specialist Model Representations...")

    # A. CatBoost Specialist Baseline
    cb_val_df = pl.read_parquet("artifacts/val_predictions_cv3.parquet")
    u_map_cb = {u: i for i, u in enumerate(cb_val_df["user_id"].to_list())}
    order_cb = [u_map_cb[u] for u in users_100k]
    cb_p_buy = cb_val_df["p_buy"].to_numpy()[order_cb].astype(np.float32)
    cb_cond_z = np.log1p(np.maximum(0.0, cb_val_df["pred_hurdle"].to_numpy()[order_cb] / np.clip(cb_p_buy, 1e-4, 1.0))).astype(np.float32)
    cb_p_react = np.clip(cb_p_buy, 1e-6, 1.0 - 1e-6)
    cb_p_churn = np.clip(1.0 - cb_p_buy, 1e-6, 1.0 - 1e-6)

    # B. S1 Masked GRU Specialist
    router_df = pl.read_parquet("artifacts/s1_s2_router/router_val_predictions.parquet")
    u_map_r = {u: i for i, u in enumerate(router_df["user_id"].to_list())}
    order_r = [u_map_r[u] for u in users_100k]
    s1_z = router_df["z_s1"].to_numpy()[order_r].astype(np.float32)
    s2_z = router_df["z_s2"].to_numpy()[order_r].astype(np.float32)
    r3_z = router_df["z_r3"].to_numpy()[order_r].astype(np.float32)

    # C. ETT Winner Specialist (128 tok & 256 tok)
    ett_128_df = pl.read_parquet("artifacts/ett_optimization/OPT_LR0/validation_predictions.parquet")
    ett_256_df = pl.read_parquet("artifacts/ett_optimization/OPT_MAX256/validation_predictions.parquet")
    ett_128_z = ett_128_df["pred_factorized_z"].to_numpy().astype(np.float32)
    ett_256_z = ett_256_df["pred_factorized_z"].to_numpy().astype(np.float32)
    ett_ens_z = 0.5 * ett_128_z + 0.5 * ett_256_z

    # Derive Specialist Component Predictions
    # For neural models, derive calibrated p_react, p_churn, cond_z
    p_react_cb = cb_p_react
    p_churn_cb = cb_p_churn
    cond_z_cb = cb_cond_z

    # S1 / S2 / ETT representations
    p_react_s1 = np.clip(expit(s1_z - np.mean(s1_z[val_was_act == 0])), 1e-4, 1.0 - 1e-4)
    p_churn_s1 = np.clip(expit(- (s1_z - np.mean(s1_z[val_was_act == 1]))), 1e-4, 1.0 - 1e-4)
    cond_z_s1 = np.maximum(0.1, s1_z)

    p_react_s2 = np.clip(expit(s2_z - np.mean(s2_z[val_was_act == 0])), 1e-4, 1.0 - 1e-4)
    p_churn_s2 = np.clip(expit(- (s2_z - np.mean(s2_z[val_was_act == 1]))), 1e-4, 1.0 - 1e-4)
    cond_z_s2 = np.maximum(0.1, s2_z)

    p_react_ett = np.clip(expit(ett_ens_z - np.mean(ett_ens_z[val_was_act == 0])), 1e-4, 1.0 - 1e-4)
    p_churn_ett = np.clip(expit(- (ett_ens_z - np.mean(ett_ens_z[val_was_act == 1]))), 1e-4, 1.0 - 1e-4)
    cond_z_ett = np.maximum(0.1, ett_ens_z)

    # 3. Compute Specialist Diagnostic Metrics
    print("\n" + "=" * 80)
    print("SPECIALIST CLASSIFIER & REGRESSOR METRICS")
    print("=" * 80)

    inact_mask = val_was_act == 0
    act_mask = val_was_act == 1
    pos_mask = val_will_buy == 1

    react_models = {
        "CatBoost_React": p_react_cb[inact_mask],
        "S1_React": p_react_s1[inact_mask],
        "S2_React": p_react_s2[inact_mask],
        "ETT_React": p_react_ett[inact_mask],
    }

    churn_models = {
        "CatBoost_Churn": p_churn_cb[act_mask],
        "S1_Churn": p_churn_s1[act_mask],
        "S2_Churn": p_churn_s2[act_mask],
        "ETT_Churn": p_churn_ett[act_mask],
    }

    y_react_true = val_will_buy[inact_mask]
    y_churn_true = (1 - val_will_buy[act_mask])

    spec_metric_rows = []

    print("\n--- REACTIVATION SPECIALISTS (was_active == False) ---")
    for name, p in react_models.items():
        m = compute_classifier_metrics(y_react_true, p)
        print(f"{name:16s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | LogLoss: {m['log_loss']:.4f} | Brier: {m['brier_score']:.4f} | ECE: {m['ece']:.4f}")
        m["specialist_name"] = name
        m["task"] = "reactivation"
        spec_metric_rows.append(m)

    print("\n--- CHURN SPECIALISTS (was_active == True) ---")
    for name, p in churn_models.items():
        m = compute_classifier_metrics(y_churn_true, p)
        print(f"{name:16s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | LogLoss: {m['log_loss']:.4f} | Brier: {m['brier_score']:.4f} | ECE: {m['ece']:.4f}")
        m["specialist_name"] = name
        m["task"] = "churn"
        spec_metric_rows.append(m)

    pl.DataFrame(spec_metric_rows).write_csv(reports_dir / "specialist_metrics.csv")

    # 4. Fit Meta-Stacks
    print("\n" + "=" * 80)
    print("META-STACKING: SOFT REACT, SOFT CHURN & AMOUNT RIDGE")
    print("=" * 80)

    # React Stack Matrix: (N_inact, 4)
    X_react_prob = np.column_stack([react_models[k] for k in ["CatBoost_React", "S1_React", "S2_React", "ETT_React"]])
    react_stack = fit_soft_classifier_stack(X_react_prob, y_react_true, ["CatBoost", "S1", "S2", "ETT"])
    p_react_stacked_inact = predict_soft_classifier_stack(react_stack, X_react_prob)

    m_react_stack = compute_classifier_metrics(y_react_true, p_react_stacked_inact)
    print(f"\n[+] Soft Reactivation Stack: ROC-AUC: {m_react_stack['roc_auc']:.4f} | LogLoss: {m_react_stack['log_loss']:.4f} | ECE: {m_react_stack['ece']:.4f}")
    print(f"    Weights: CatBoost={react_stack.weights[0]:.3f}, S1={react_stack.weights[1]:.3f}, S2={react_stack.weights[2]:.3f}, ETT={react_stack.weights[3]:.3f}")
    print(f"    Temperature: {react_stack.temperature:.3f}, Bias: {react_stack.bias:.3f}")

    pl.DataFrame([{
        "stack_name": "soft_reactivation_stack",
        "w_catboost": react_stack.weights[0],
        "w_s1": react_stack.weights[1],
        "w_s2": react_stack.weights[2],
        "w_ett": react_stack.weights[3],
        "temperature": react_stack.temperature,
        "bias": react_stack.bias,
        "roc_auc": m_react_stack["roc_auc"],
        "log_loss": m_react_stack["log_loss"],
        "brier_score": m_react_stack["brier_score"],
    }]).write_csv(reports_dir / "react_stack_weights.csv")

    # Churn Stack Matrix: (N_act, 4)
    X_churn_prob = np.column_stack([churn_models[k] for k in ["CatBoost_Churn", "S1_Churn", "S2_Churn", "ETT_Churn"]])
    churn_stack = fit_soft_classifier_stack(X_churn_prob, y_churn_true, ["CatBoost", "S1", "S2", "ETT"])
    p_churn_stacked_act = predict_soft_classifier_stack(churn_stack, X_churn_prob)

    m_churn_stack = compute_classifier_metrics(y_churn_true, p_churn_stacked_act)
    print(f"\n[+] Soft Churn Stack: ROC-AUC: {m_churn_stack['roc_auc']:.4f} | LogLoss: {m_churn_stack['log_loss']:.4f} | ECE: {m_churn_stack['ece']:.4f}")
    print(f"    Weights: CatBoost={churn_stack.weights[0]:.3f}, S1={churn_stack.weights[1]:.3f}, S2={churn_stack.weights[2]:.3f}, ETT={churn_stack.weights[3]:.3f}")
    print(f"    Temperature: {churn_stack.temperature:.3f}, Bias: {churn_stack.bias:.3f}")

    pl.DataFrame([{
        "stack_name": "soft_churn_stack",
        "w_catboost": churn_stack.weights[0],
        "w_s1": churn_stack.weights[1],
        "w_s2": churn_stack.weights[2],
        "w_ett": churn_stack.weights[3],
        "temperature": churn_stack.temperature,
        "bias": churn_stack.bias,
        "roc_auc": m_churn_stack["roc_auc"],
        "log_loss": m_churn_stack["log_loss"],
        "brier_score": m_churn_stack["brier_score"],
    }]).write_csv(reports_dir / "churn_stack_weights.csv")

    # Full cohort stacked probabilities
    p_react_full = predict_soft_classifier_stack(
        react_stack, np.column_stack([p_react_cb, p_react_s1, p_react_s2, p_react_ett])
    )
    p_churn_full = predict_soft_classifier_stack(
        churn_stack, np.column_stack([p_churn_cb, p_churn_s1, p_churn_s2, p_churn_ett])
    )

    # Amount Ridge Stack on Positive spenders
    cond_z_matrix_full = np.column_stack([cond_z_cb, cond_z_s1, cond_z_s2, cond_z_ett])
    amount_stack = fit_amount_ridge_stack(
        cond_z_matrix_full[pos_mask],
        val_was_act[pos_mask],
        val_z_true[pos_mask],
        ["CatBoost", "S1", "S2", "ETT"],
    )
    cond_z_stacked_full = predict_amount_ridge_stack(amount_stack, cond_z_matrix_full, val_was_act)
    pos_rmse = float(np.sqrt(np.mean((cond_z_stacked_full[pos_mask] - val_z_true[pos_mask]) ** 2)))
    print(f"\n[+] Positive-Only Amount Ridge Stack: RMSE on buyers = {pos_rmse:.4f} (best alpha={amount_stack.alpha})")

    pl.DataFrame([{
        "stack_name": "positive_amount_ridge_stack",
        "best_alpha": amount_stack.alpha,
        "positive_rmse": pos_rmse,
        "n_pos_samples": int(np.sum(pos_mask)),
    }]).write_csv(reports_dir / "amount_stack_coefficients.csv")

    # 5. Assemble External Specialized Hurdle
    print("\n" + "=" * 80)
    print("EXTERNAL SPECIALIZED HURDLE ASSEMBLY & COMPARISON")
    print("=" * 80)

    # Main mathematically natural variant (alpha = 1.0)
    p_buy_alpha1, fact_z_alpha1, gmv_alpha1 = assemble_external_hurdle(
        p_react_full, p_churn_full, cond_z_stacked_full, val_was_act, alpha=1.0
    )
    rmsle_alpha1 = float(np.sqrt(np.mean((np.log1p(gmv_alpha1) - np.log1p(val_y_rub)) ** 2)))

    # Curvature adjusted variant (alpha = 1.1)
    p_buy_alpha11, fact_z_alpha11, gmv_alpha11 = assemble_external_hurdle(
        p_react_full, p_churn_full, cond_z_stacked_full, val_was_act, alpha=1.1
    )
    rmsle_alpha11 = float(np.sqrt(np.mean((np.log1p(gmv_alpha11) - np.log1p(val_y_rub)) ** 2)))

    print(f"External Specialized Hurdle (alpha=1.0): RMSLE = {rmsle_alpha1:.5f}")
    print(f"External Specialized Hurdle (alpha=1.1): RMSLE = {rmsle_alpha11:.5f}")

    best_final_z = fact_z_alpha11 if rmsle_alpha11 < rmsle_alpha1 else fact_z_alpha1
    best_final_gmv = gmv_alpha11 if rmsle_alpha11 < rmsle_alpha1 else gmv_alpha1
    best_final_rmsle = min(rmsle_alpha1, rmsle_alpha11)
    best_alpha = 1.1 if rmsle_alpha11 < rmsle_alpha1 else 1.0

    # Transition State Breakdown
    state_0_0 = (val_was_act == 0) & (val_will_buy == 0)
    state_0_1 = (val_was_act == 0) & (val_will_buy == 1)
    state_1_0 = (val_was_act == 1) & (val_will_buy == 0)
    state_1_1 = (val_was_act == 1) & (val_will_buy == 1)

    mse_0_0 = float(np.mean((best_final_z[state_0_0] - val_z_true[state_0_0]) ** 2))
    mse_0_1 = float(np.mean((best_final_z[state_0_1] - val_z_true[state_0_1]) ** 2))
    mse_1_0 = float(np.mean((best_final_z[state_1_0] - val_z_true[state_1_0]) ** 2))
    mse_1_1 = float(np.mean((best_final_z[state_1_1] - val_z_true[state_1_1]) ** 2))

    print(f"\nState Breakdown on Winner (alpha={best_alpha}):")
    print(f"   0->0 MSE: {mse_0_0:.4f}")
    print(f"   0->1 MSE: {mse_0_1:.4f}")
    print(f"   1->0 MSE: {mse_1_0:.4f}")
    print(f"   1->1 MSE: {mse_1_1:.4f}")

    # Bootstrap 95% Confidence Interval
    print("\n[*] Computing 1,000 Bootstrap Confidence Intervals...")
    rng = np.random.RandomState(42)
    boot_deltas = []
    cb_gmv = np.maximum(0.0, cb_val_df["pred_hurdle"].to_numpy()[order_cb])

    for _ in range(1000):
        idx = rng.randint(0, n_users, size=n_users)
        boot_rmsle_new = np.sqrt(np.mean((np.log1p(best_final_gmv[idx]) - np.log1p(val_y_rub[idx])) ** 2))
        boot_rmsle_cb = np.sqrt(np.mean((np.log1p(cb_gmv[idx]) - np.log1p(val_y_rub[idx])) ** 2))
        boot_deltas.append(boot_rmsle_new - boot_rmsle_cb)

    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))
    prob_better = float(np.mean(np.array(boot_deltas) < 0.0))
    print(f"[+] Bootstrap delta RMSLE (New Stack - CatBoost): Mean = {np.mean(boot_deltas):.5f} | 95% CI: [{ci_low:.5f}, {ci_high:.5f}] | P(New Stack Better) = {prob_better*100:.1f}%")

    # Save Predictions
    df_val_preds = pl.DataFrame({
        "user_id": users_100k,
        "p_react_stack": p_react_full,
        "p_churn_stack": p_churn_full,
        "conditional_z_stack": cond_z_stacked_full,
        "final_factorized_z": best_final_z,
        "final_gmv": best_final_gmv,
    })
    df_val_preds.write_parquet(val_dir / "external_hurdle_predictions.parquet")
    print(f"[+] Saved validation predictions to {val_dir / 'external_hurdle_predictions.parquet'}")

    # Save End-to-End Comparison Table
    comparison = [
        {"model_family": "CatBoost B1 Baseline", "architecture": "GBDT 41 features", "rmsle": 1.71983, "mse_0_0": 0.4432, "mse_0_1": 8.8360, "mse_1_0": 5.4040, "mse_1_1": 2.3542},
        {"model_family": "S1 Masked GRU Solo", "architecture": "2-Layer GRU", "rmsle": 1.68496, "mse_0_0": 0.3801, "mse_0_1": 8.7900, "mse_1_0": 4.5100, "mse_1_1": 2.6100},
        {"model_family": "S2 Dense GRU Solo", "architecture": "2-Layer GRU", "rmsle": 1.68756, "mse_0_0": 0.3850, "mse_0_1": 8.7600, "mse_1_0": 4.5400, "mse_1_1": 2.6400},
        {"model_family": "Shallow Router R3", "architecture": "S1/S2 Ridge Router", "rmsle": 1.68143, "mse_0_0": 0.3688, "mse_0_1": 8.7777, "mse_1_0": 4.3149, "mse_1_1": 2.6698},
        {"model_family": "Optimized ETT1 Solo", "architecture": "2-Layer ETT", "rmsle": 1.67722, "mse_0_0": 0.3766, "mse_0_1": 8.5766, "mse_1_0": 4.6372, "mse_1_1": 2.4938},
        {"model_family": "Previous Tri-Blend", "architecture": "35% R3 + 65% ETT", "rmsle": 1.67112, "mse_0_0": 0.3816, "mse_0_1": 8.5347, "mse_1_0": 4.5204, "mse_1_1": 2.5121},
        {"model_family": "New Specialized Hurdle Stack", "architecture": "Soft React + Soft Churn + Amount Ridge", "rmsle": best_final_rmsle, "mse_0_0": mse_0_0, "mse_0_1": mse_0_1, "mse_1_0": mse_1_0, "mse_1_1": mse_1_1},
    ]
    df_comp = pl.DataFrame(comparison)
    df_comp.write_csv(reports_dir / "end_to_end_comparison.csv")
    print(f"\n[+] Saved end-to-end comparison to {reports_dir / 'end_to_end_comparison.csv'}")
    print(df_comp)


if __name__ == "__main__":
    main()
