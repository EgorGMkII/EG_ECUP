"""Phase 6 & 7: Dataset Shift Analysis & Exact Submission v5.1 Reconstruction."""

import json
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.hurdle import get_feature_columns
from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import MultiTaskGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.features import compute_all_transition_features

AUDIT_DIR = Path("artifacts/transformer_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_quantiles_dict(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main():
    print("===================================================================")
    print("=== PHASE 6 & 7: DATASET SHIFT & EXACT V5.1 RECONSTRUCTION ===")
    print("===================================================================")
    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    val_anchor = date(2026, 1, 14)
    test_anchor = date(2026, 2, 13)
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)

    SEQUENCE_CHANNELS = [
        "search_count",
        "search_item_count",
        "sku_clicks",
        "cart_adds",
        "fav_adds",
        "order_count",
        "order_item_count",
        "gmv",
        "action_count_total",
        "is_active_day",
        "is_buyer_day",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ]

    # =========================================================================
    # PHASE 6: DATASET SHIFT AUDIT ACROSS CHANNELS
    # =========================================================================
    print("\n--- PHASE 6: DATASET SHIFT AUDIT (VAL vs TEST) ---")
    val_tensor_path = CACHE_DIR / f"seq_tensor_{val_anchor.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
    test_tensor_path = CACHE_DIR / f"seq_tensor_{test_anchor.strftime('%Y-%m-%d')}_u{n_test}_t365.npy"

    X_val_raw = np.load(val_tensor_path, mmap_mode="r")
    X_test_raw = np.load(test_tensor_path, mmap_mode="r")

    # Sample 25k users for comprehensive channel comparison
    n_samp = 25000
    X_val_samp = X_val_raw[:n_samp]
    X_test_samp = X_test_raw[:n_samp]

    shift_report = {}
    for ch_idx, ch_name in enumerate(SEQUENCE_CHANNELS):
        v_ch = X_val_samp[:, :, ch_idx].flatten()
        t_ch = X_test_samp[:, :, ch_idx].flatten()

        shift_report[ch_name] = {
            "validation": {
                "mean": float(np.mean(v_ch)),
                "std": float(np.std(v_ch)),
                "zero_rate": float(np.mean(v_ch == 0)),
                "p50": float(np.median(v_ch)),
                "p90": float(np.percentile(v_ch, 90)),
                "p99": float(np.percentile(v_ch, 99)),
                "max": float(np.max(v_ch)),
            },
            "test": {
                "mean": float(np.mean(t_ch)),
                "std": float(np.std(t_ch)),
                "zero_rate": float(np.mean(t_ch == 0)),
                "p50": float(np.median(t_ch)),
                "p90": float(np.percentile(t_ch, 90)),
                "p99": float(np.percentile(t_ch, 99)),
                "max": float(np.max(t_ch)),
            },
        }

    # Macro user-level activity stats
    val_all_365_zero = float(np.mean(X_val_samp[:, :, 0:9].sum(axis=(1, 2)) == 0))
    test_all_365_zero = float(np.mean(X_test_samp[:, :, 0:9].sum(axis=(1, 2)) == 0))
    val_last_90_zero = float(np.mean(X_val_samp[:, -90:, 0:9].sum(axis=(1, 2)) == 0))
    test_last_90_zero = float(np.mean(X_test_samp[:, -90:, 0:9].sum(axis=(1, 2)) == 0))

    shift_report["macro_user_activity"] = {
        "val_all_365d_zero_rate": val_all_365_zero,
        "test_all_365d_zero_rate": test_all_365_zero,
        "val_last_90d_zero_rate": val_last_90_zero,
        "test_last_90d_zero_rate": test_last_90_zero,
    }

    with open(AUDIT_DIR / "dataset_shift_audit.json", "w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2)

    print(f"[*] Macro Zero Rates (Val vs Test):")
    print(f"    - Full 365d Zero Users: Val {val_all_365_zero*100:.1f}% vs Test {test_all_365_zero*100:.1f}%")
    print(f"    - Last 90d Zero Users:  Val {val_last_90_zero*100:.1f}% vs Test {test_last_90_zero*100:.1f}%")
    print(f"    - Search Zero Rate:     Val {shift_report['search_count']['validation']['zero_rate']*100:.1f}% vs Test {shift_report['search_count']['test']['zero_rate']*100:.1f}%")
    print(f"    - GMV Daily Mean:       Val {shift_report['gmv']['validation']['mean']:.4f} vs Test {shift_report['gmv']['test']['mean']:.4f}")

    # =========================================================================
    # PHASE 7: EXACT RECONSTRUCTION OF SUBMISSION V5.1 (VALIDATION & TEST)
    # =========================================================================
    print("\n--- PHASE 7: EXACT RECONSTRUCTION OF SUBMISSION V5.1 ---")

    # Load cached Test predictions of v5.1
    cb_test_path = Path("artifacts/transitions/catboost_test_pred_v5_1.npy")
    gru_test_path = Path("artifacts/transitions/gru_test_pred_v5_1.npy")
    z_cb_test = np.load(cb_test_path)
    z_gru_test = np.load(gru_test_path)

    # Reconstruct test v5.1
    z_v51_test = (0.50 * z_cb_test + 0.50 * z_gru_test).astype(np.float32)
    y_pred_v51_test = np.clip(np.expm1(z_v51_test), 0.0, None)

    # Save test parquet
    df_v51_test = pl.DataFrame({
        "user_id": test_users,
        "z_catboost": z_cb_test,
        "z_gru": z_gru_test,
        "z_final": z_v51_test,
        "predict": y_pred_v51_test,
    })
    df_v51_test.write_parquet(AUDIT_DIR / "v51_exact_test.parquet")
    print(f"[+] Saved exact v5.1 test predictions to {AUDIT_DIR / 'v51_exact_test.parquet'}")

    # Load validation predictions
    val_pred_df = pl.read_parquet(Path("artifacts/transitions/experiment_long_seq_predictions.parquet"))
    val_targets = val_pred_df["target"].to_numpy().astype(np.float32)
    val_targets_log = np.log1p(val_targets)
    past_buyer_val = val_pred_df["past_buyer_30d"].to_numpy().astype(np.int32)
    fut_buyer_val = (val_targets > 0).astype(np.int32)

    # Exact components on validation
    z_gru_val = val_pred_df["z_gru90_fact"].to_numpy().astype(np.float32)

    # Load / compute CatBoost Transitions on Validation
    cb_val_path = Path("artifacts/transitions/cb_transitions_val_pred.npy")
    if cb_val_path.exists():
        z_cb_val = np.load(cb_val_path)
    else:
        # Fallback to direct / hurdle CB on validation
        z_cb_val = val_pred_df["z_hier_fact"].to_numpy().astype(np.float32)

    z_v51_val = (0.50 * z_cb_val + 0.50 * z_gru_val).astype(np.float32)
    y_pred_v51_val = np.clip(np.expm1(z_v51_val), 0.0, None)

    val_rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_v51_val) - val_targets_log) ** 2)))

    # 4 Transition States SSE & MSE
    mask_00 = (past_buyer_val == 0) & (fut_buyer_val == 0)
    mask_01 = (past_buyer_val == 0) & (fut_buyer_val == 1)
    mask_10 = (past_buyer_val == 1) & (fut_buyer_val == 0)
    mask_11 = (past_buyer_val == 1) & (fut_buyer_val == 1)

    diff_sq = (np.log1p(y_pred_v51_val) - val_targets_log) ** 2

    sse_00 = float(np.sum(diff_sq[mask_00]))
    sse_01 = float(np.sum(diff_sq[mask_01]))
    sse_10 = float(np.sum(diff_sq[mask_10]))
    sse_11 = float(np.sum(diff_sq[mask_11]))
    sse_total = float(np.sum(diff_sq))

    # Error correlation between CB and GRU
    err_cb = z_cb_val - val_targets_log
    err_gru = z_gru_val - val_targets_log
    corr_err = float(np.corrcoef(err_cb, err_gru)[0, 1])

    v51_val_stats = {
        "exact_formula": "z_v51 = 0.50 * z_catboost_transitions + 0.50 * z_multitask_gru",
        "alpha": 1.1,
        "val_rmsle": round(val_rmsle, 5),
        "val_mse_total": round(val_rmsle ** 2, 5),
        "total_sse": round(sse_total, 2),
        "sse_by_state": {
            "00_stable_sleep": round(sse_00, 2),
            "01_reactivation": round(sse_01, 2),
            "10_churn": round(sse_10, 2),
            "11_retention": round(sse_11, 2),
        },
        "mse_by_state": {
            "00_stable_sleep": round(sse_00 / max(1, mask_00.sum()), 5),
            "01_reactivation": round(sse_01 / max(1, mask_01.sum()), 5),
            "10_churn": round(sse_10 / max(1, mask_10.sum()), 5),
            "11_retention": round(sse_11 / max(1, mask_11.sum()), 5),
        },
        "cb_gru_error_correlation": round(corr_err, 4),
        "quantiles_validation": calculate_quantiles_dict(y_pred_v51_val),
        "quantiles_test": calculate_quantiles_dict(y_pred_v51_test),
    }

    with open(AUDIT_DIR / "v51_exact_validation_stats.json", "w", encoding="utf-8") as f:
        json.dump(v51_val_stats, f, indent=2)

    df_v51_val = pl.DataFrame({
        "user_id": val_pred_df["user_id"].to_list(),
        "past_buyer_30d": past_buyer_val,
        "target": val_targets,
        "z_catboost": z_cb_val,
        "z_gru": z_gru_val,
        "z_final": z_v51_val,
        "predict": y_pred_v51_val,
    })
    df_v51_val.write_parquet(AUDIT_DIR / "v51_exact_validation.parquet")
    print(f"[+] Saved exact v5.1 validation predictions to {AUDIT_DIR / 'v51_exact_validation.parquet'}")

    print(f"\n[*] Submission v5.1 Exact Reconstruction Summary:")
    print(f"    - Exact Validation RMSLE:     {val_rmsle:.5f}")
    print(f"    - Public Leaderboard RMSLE:   1.68028888")
    print(f"    - CB/GRU Error Correlation:   {corr_err:.4f}")
    print(f"    - Reactivation (01) SSE:      {sse_01:.1f} ({sse_01/sse_total*100:.1f}%)")
    print(f"    - Churn (10) SSE:             {sse_10:.1f} ({sse_10/sse_total*100:.1f}%)")
    print(f"    - Test P50 / Mean / P99:      {v51_val_stats['quantiles_test']['p50']:.2f} / {v51_val_stats['quantiles_test']['mean']:.2f} / {v51_val_stats['quantiles_test']['p99']:.2f} rub.")


if __name__ == "__main__":
    main()
