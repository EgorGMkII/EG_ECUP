"""Master Autonomous Orchestrator for MultiTask GRU Experimental Sweep (Stages 0 to 5)."""

import gc
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, os.getcwd())
if "/job" not in sys.path:
    sys.path.insert(0, "/job")

import numpy as np
import polars as pl
import torch

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.sequential.gru_sweep import (
    get_anchor_set,
    extract_or_load_master_sequence_tensor,
    SlicedMemmapDataset,
)
from src.sequential.gru_experiment_runner import run_gru_training_and_eval
from src.sequential.gru_logging import ARTIFACTS_ROOT, initialize_registry


def main():
    print("===================================================================")
    print("=== AUTONOMOUS MULTITASK GRU SYSTEMATIC EXPERIMENTAL SWEEP ===")
    print("===================================================================")
    t_sweep_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_registry()

    # 1. Base Data & Users
    print("[*] Loading raw panel data and 100,000 selected user subset...")
    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    val_anchor = date(2026, 1, 14)

    # 2. Validation Targets and Past Buyer Status
    print(f"[*] Preparing Validation dataset for anchor {val_anchor}...")
    y_val = extract_anchor_targets(data, train_sample_users, val_anchor)
    snap_val_path = Path(f"data/snapshots/snapshot_{val_anchor.strftime('%Y-%m-%d')}.parquet")
    if snap_val_path.exists():
        snap_val = pl.read_parquet(snap_val_path)
        past_gmv_val = snap_val["gmv_sum_30d"].to_numpy().astype(np.float32)
        past_buyer_val = (past_gmv_val > 0).astype(np.int32)
        del snap_val
    else:
        w_val_start = val_anchor - timedelta(days=30)
        gmv_val_agg = (
            data.filter((pl.col("event_date") > w_val_start) & (pl.col("event_date") <= val_anchor))
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("gmv_sum"))
        )
        user_val_df = pl.DataFrame({"user_id": train_sample_users})
        joined_val = user_val_df.join(gmv_val_agg, on="user_id", how="left").fill_null(0.0)
        past_buyer_val = (joined_val["gmv_sum"].to_numpy().astype(np.float32) > 0).astype(np.int32)

    # 3. Validation Raw 180d Tensor and Scaler
    val_tensor_180_raw = extract_or_load_master_sequence_tensor(data, train_sample_users, val_anchor, max_seq_len=180, include_is_observed=False)
    scaler_180 = SequentialScaler().fit(val_tensor_180_raw[:25000])

    # 4. CatBoost Validation Predictions (Fixed Reference for 50/50 Blends)
    val_cb_path = Path("artifacts/transformer_audit/v51_exact_validation.parquet")
    if val_cb_path.exists():
        df_v51_val = pl.read_parquet(val_cb_path)
        val_catboost_z = df_v51_val["z_catboost"].to_numpy().astype(np.float32)
    else:
        cb_val_path = Path("artifacts/transitions/cb_transitions_val_pred.npy")
        if cb_val_path.exists():
            val_catboost_z = np.load(cb_val_path)
        else:
            val_catboost_z = np.zeros_like(y_val, dtype=np.float32)

    # Cache helper for training tensors
    def get_dataset_for_anchors(anchor_dates: list, seq_len: int, include_obs: bool = False) -> SlicedMemmapDataset:
        train_paths, train_targets, past_buyers = [], [], []
        for a in anchor_dates:
            # Load master 180d tensor
            t_mmap = extract_or_load_master_sequence_tensor(data, train_sample_users, a, max_seq_len=180, include_is_observed=include_obs)
            obs_suffix = "_obs" if include_obs else ""
            filename = f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t180{obs_suffix}.npy"
            train_paths.append(CACHE_DIR / filename)

            # Targets & past buyer
            train_targets.append(extract_anchor_targets(data, train_sample_users, a))
            snap_path = Path(f"data/snapshots/snapshot_{a.strftime('%Y-%m-%d')}.parquet")
            if snap_path.exists():
                snap_a = pl.read_parquet(snap_path)
                past_buyers.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
                del snap_a
            else:
                w_start = a - timedelta(days=30)
                gmv_agg = (
                    data.filter((pl.col("event_date") > w_start) & (pl.col("event_date") <= a))
                    .group_by("user_id")
                    .agg(pl.col("gmv").sum().alias("gmv_sum"))
                )
                user_df = pl.DataFrame({"user_id": train_sample_users})
                joined = user_df.join(gmv_agg, on="user_id", how="left").fill_null(0.0)
                past_buyers.append((joined["gmv_sum"].to_numpy().astype(np.float32) > 0).astype(np.int32))

        # Scaler
        cur_scaler = SequentialScaler().fit(np.load(train_paths[-1], mmap_mode="r")[:25000])
        return SlicedMemmapDataset(train_paths, train_targets, past_buyers, seq_len=seq_len, scaler=cur_scaler), cur_scaler

    def run_or_load_experiment(cfg, ds, val_tensor, cur_scaler):
        metrics_file = ARTIFACTS_ROOT / cfg["run_id"] / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("status") == "completed":
                print(f"[+] [CACHED] Run {cfg['run_id']} already completed | Factorized RMSLE: {saved['rmsle_factorized']:.5f} | Blend RMSLE: {saved['rmsle_blend_cb']:.5f}")
                return saved
        return run_gru_training_and_eval(
            cfg, ds, val_tensor, y_val, past_buyer_val, train_sample_users, val_catboost_z, cur_scaler, device
        )

    # =========================================================================
    # STAGE 0: BASELINE VERIFICATION (MultiTask GRU v5.1)
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE 0] BASELINE REPRODUCTION (MultiTask GRU v5.1) ===")
    print("===================================================================")

    recent_14_anchors = get_anchor_set("recent_14")
    ds_stage0, scaler_stage0 = get_dataset_for_anchors(recent_14_anchors, seq_len=90, include_obs=False)

    cfg_stage0 = {
        "run_id": "gru_baseline_v51_rep",
        "sequence_length": 90,
        "anchor_set": "recent_14",
        "n_anchors": 14,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 512,
        "epochs": 8,
        "lambda_cls": 0.5,
        "lambda_reg": 1.0,
        "alpha": 1.1,
        "seed": 42,
        "notes": "Stage 0: Baseline verification against v5.1 exact RMSLE",
    }

    res_stage0 = run_or_load_experiment(
        cfg_stage0, ds_stage0, val_tensor_180_raw, scaler_stage0
    )

    baseline_rmsle = res_stage0["rmsle_factorized"]
    baseline_blend = res_stage0["rmsle_blend_cb"]
    print(f"\n[*] Baseline Result: Factorized RMSLE = {baseline_rmsle:.5f} | Blend RMSLE = {baseline_blend:.5f}")

    # =========================================================================
    # STAGE A: SEQUENCE LENGTH SWEEP (L44, L60, L90, L120, L180)
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE A] SEQUENCE LENGTH SWEEP (L44 to L180) ===")
    print("===================================================================")

    stage_a_lengths = [44, 60, 90, 120, 180]
    stage_a_results = []

    for L in stage_a_lengths:
        if L == 90:
            stage_a_results.append(res_stage0)
            continue

        print(f"\n[*] Running Sequence Length L = {L} days...")
        ds_l, scaler_l = get_dataset_for_anchors(recent_14_anchors, seq_len=L, include_obs=False)
        cfg_l = {
            "run_id": f"gru_len_L{L}_recent14",
            "sequence_length": L,
            "anchor_set": "recent_14",
            "n_anchors": 14,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 8,
            "lambda_cls": 0.5,
            "lambda_reg": 1.0,
            "alpha": 1.1,
            "seed": 42,
            "notes": f"Stage A: Sequence length sweep L={L}",
        }
        res_l = run_or_load_experiment(
            cfg_l, ds_l, val_tensor_180_raw, scaler_l
        )
        stage_a_results.append(res_l)

    # Rank Stage A by Blend RMSLE
    stage_a_sorted = sorted(stage_a_results, key=lambda x: (x["rmsle_blend_cb"], x["rmsle_factorized"]))
    best_len_1 = stage_a_sorted[0]["sequence_length"]
    best_len_2 = stage_a_sorted[1]["sequence_length"]

    print(f"\n[+] STAGE A COMPLETE. Top 2 Lengths: L1 = {best_len_1}d (Blend: {stage_a_sorted[0]['rmsle_blend_cb']:.5f}), L2 = {best_len_2}d (Blend: {stage_a_sorted[1]['rmsle_blend_cb']:.5f})")

    # =========================================================================
    # STAGE B: ANCHOR COVERAGE (all_existing_22 vs recent_14)
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE B] ANCHOR COVERAGE SWEEP (all_existing_22) ===")
    print("===================================================================")

    all_22_anchors = get_anchor_set("all_existing_22")
    stage_b_results = []

    for L in [best_len_1, best_len_2]:
        print(f"\n[*] Training L = {L} on all_existing_22 anchors ({len(all_22_anchors)} panels)...")
        ds_22, scaler_22 = get_dataset_for_anchors(all_22_anchors, seq_len=L, include_obs=False)
        cfg_22 = {
            "run_id": f"gru_len_L{L}_all22",
            "sequence_length": L,
            "anchor_set": "all_existing_22",
            "n_anchors": len(all_22_anchors),
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 6,  # 2.2M samples = more steps per epoch
            "lambda_cls": 0.5,
            "lambda_reg": 1.0,
            "alpha": 1.1,
            "seed": 42,
            "notes": f"Stage B: Anchor coverage all_existing_22 for L={L}",
        }
        res_22 = run_or_load_experiment(
            cfg_22, ds_22, val_tensor_180_raw, scaler_22
        )
        stage_b_results.append(res_22)

    # =========================================================================
    # STAGE C: FULL CALENDAR & SPRING ANCHOR (2025-02-13)
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE C] FULL CALENDAR & EARLY SPRING ANCHOR (2025-02-13) ===")
    print("===================================================================")

    full_cal_anchors = get_anchor_set("full_calendar")
    print(f"[*] Full calendar anchors ({len(full_cal_anchors)} panels), including 2025-02-13...")

    stage_c_results = []
    # Validation tensor with 16th channel (is_observed)
    val_tensor_obs_raw = extract_or_load_master_sequence_tensor(data, train_sample_users, val_anchor, max_seq_len=180, include_is_observed=True)

    for L in [best_len_1, best_len_2]:
        print(f"\n[*] Training L = {L} on full_calendar anchors with is_observed channel...")
        ds_cal, scaler_cal = get_dataset_for_anchors(full_cal_anchors, seq_len=L, include_obs=True)
        cfg_cal = {
            "run_id": f"gru_len_L{L}_full_calendar",
            "sequence_length": L,
            "anchor_set": "full_calendar",
            "n_anchors": len(full_cal_anchors),
            "input_dim": 16,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": 0.5,
            "lambda_reg": 1.0,
            "alpha": 1.1,
            "seed": 42,
            "notes": f"Stage C: Full calendar with 2025-02-13 spring anchor and left-padding for L={L}",
        }
        res_cal = run_or_load_experiment(
            cfg_cal, ds_cal, val_tensor_obs_raw, scaler_cal
        )
        stage_c_results.append(res_cal)

    # Determine Winning Dataset Configuration
    all_dataset_runs = stage_a_results + stage_b_results + stage_c_results
    best_dataset_run = sorted(all_dataset_runs, key=lambda x: (x["rmsle_blend_cb"], x["rmsle_factorized"]))[0]
    win_len = best_dataset_run["sequence_length"]
    win_anchors_name = best_dataset_run["anchor_set"]
    win_input_dim = 16 if win_anchors_name == "full_calendar" else 15
    win_val_tensor = val_tensor_obs_raw if win_input_dim == 16 else val_tensor_180_raw

    print(f"\n[+] STAGE C COMPLETE. Winning Dataset Configuration: L = {win_len}d | Anchors = {win_anchors_name} | Blend RMSLE = {best_dataset_run['rmsle_blend_cb']:.5f}")

    # =========================================================================
    # STAGE D: LIMITED HYPERPARAMETER REFINEMENT
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE D] HYPERPARAMETER REFINEMENT ON WINNING DATASET ===")
    print("===================================================================")

    win_anchors = get_anchor_set(win_anchors_name)
    ds_win, scaler_win = get_dataset_for_anchors(win_anchors, seq_len=win_len, include_obs=(win_input_dim == 16))

    # Base winning hyperparams
    cur_hparams = {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "lambda_cls": 0.5,
    }

    # D1: Hidden Size (64, 96, 128)
    print("\n--- [D1] Hidden Size Sweep (64, 96, 128) ---")
    d1_runs = []
    for h in [64, 96, 128]:
        if h == 128:
            d1_runs.append(best_dataset_run)
            continue
        cfg_d1 = {
            "run_id": f"gru_tune_h{h}_{win_anchors_name}_L{win_len}",
            "sequence_length": win_len,
            "anchor_set": win_anchors_name,
            "input_dim": win_input_dim,
            "hidden_size": h,
            "num_layers": cur_hparams["num_layers"],
            "dropout": cur_hparams["dropout"],
            "learning_rate": cur_hparams["learning_rate"],
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": cur_hparams["lambda_cls"],
            "seed": 42,
            "notes": f"Stage D1: Hidden size {h}",
        }
        res_d1 = run_or_load_experiment(cfg_d1, ds_win, win_val_tensor, scaler_win)
        d1_runs.append(res_d1)

    best_h = sorted(d1_runs, key=lambda x: x["rmsle_blend_cb"])[0]["hidden_size"]
    cur_hparams["hidden_size"] = best_h
    print(f"[+] D1 Winner: hidden_size = {best_h}")

    # D2: Num Layers (1 vs 2)
    print("\n--- [D2] Num Layers Sweep (1 vs 2) ---")
    d2_runs = []
    for nl in [1, 2]:
        if nl == 2:
            d2_runs.append(sorted(d1_runs, key=lambda x: x["rmsle_blend_cb"])[0])
            continue
        cfg_d2 = {
            "run_id": f"gru_tune_layers{nl}_h{best_h}_{win_anchors_name}",
            "sequence_length": win_len,
            "anchor_set": win_anchors_name,
            "input_dim": win_input_dim,
            "hidden_size": best_h,
            "num_layers": nl,
            "dropout": cur_hparams["dropout"],
            "learning_rate": cur_hparams["learning_rate"],
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": cur_hparams["lambda_cls"],
            "seed": 42,
            "notes": f"Stage D2: Layers {nl}",
        }
        res_d2 = run_or_load_experiment(cfg_d2, ds_win, win_val_tensor, scaler_win)
        d2_runs.append(res_d2)

    best_nl = sorted(d2_runs, key=lambda x: x["rmsle_blend_cb"])[0]["num_layers"]
    cur_hparams["num_layers"] = best_nl
    print(f"[+] D2 Winner: num_layers = {best_nl}")

    # D3: Dropout (0.0, 0.15, 0.30)
    print("\n--- [D3] Dropout Sweep (0.0, 0.15, 0.30) ---")
    d3_runs = []
    for dp in [0.0, 0.15, 0.30]:
        if dp == 0.15:
            d3_runs.append(sorted(d2_runs, key=lambda x: x["rmsle_blend_cb"])[0])
            continue
        cfg_d3 = {
            "run_id": f"gru_tune_dp{int(dp*100)}_h{best_h}_l{best_nl}",
            "sequence_length": win_len,
            "anchor_set": win_anchors_name,
            "input_dim": win_input_dim,
            "hidden_size": best_h,
            "num_layers": best_nl,
            "dropout": dp,
            "learning_rate": cur_hparams["learning_rate"],
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": cur_hparams["lambda_cls"],
            "seed": 42,
            "notes": f"Stage D3: Dropout {dp}",
        }
        res_d3 = run_or_load_experiment(cfg_d3, ds_win, win_val_tensor, scaler_win)
        d3_runs.append(res_d3)

    best_dp = sorted(d3_runs, key=lambda x: x["rmsle_blend_cb"])[0]["dropout"]
    cur_hparams["dropout"] = best_dp
    print(f"[+] D3 Winner: dropout = {best_dp}")

    # D4: Learning Rate (5e-4, 1e-3, 2e-3)
    print("\n--- [D4] Learning Rate Sweep (5e-4, 1e-3, 2e-3) ---")
    d4_runs = []
    for lr_val in [5e-4, 1e-3, 2e-3]:
        if lr_val == 1e-3:
            d4_runs.append(sorted(d3_runs, key=lambda x: x["rmsle_blend_cb"])[0])
            continue
        cfg_d4 = {
            "run_id": f"gru_tune_lr{lr_val}_h{best_h}",
            "sequence_length": win_len,
            "anchor_set": win_anchors_name,
            "input_dim": win_input_dim,
            "hidden_size": best_h,
            "num_layers": best_nl,
            "dropout": best_dp,
            "learning_rate": lr_val,
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": cur_hparams["lambda_cls"],
            "seed": 42,
            "notes": f"Stage D4: LR {lr_val}",
        }
        res_d4 = run_or_load_experiment(cfg_d4, ds_win, win_val_tensor, scaler_win)
        d4_runs.append(res_d4)

    best_lr = sorted(d4_runs, key=lambda x: x["rmsle_blend_cb"])[0]["learning_rate"]
    cur_hparams["learning_rate"] = best_lr
    print(f"[+] D4 Winner: learning_rate = {best_lr}")

    # D5: Transition Loss Weight (0.25, 0.50, 1.00)
    print("\n--- [D5] Transition Loss Weight Sweep (0.25, 0.50, 1.00) ---")
    d5_runs = []
    for l_cls in [0.25, 0.50, 1.00]:
        if l_cls == 0.50:
            d5_runs.append(sorted(d4_runs, key=lambda x: x["rmsle_blend_cb"])[0])
            continue
        cfg_d5 = {
            "run_id": f"gru_tune_lcls{int(l_cls*100)}_h{best_h}",
            "sequence_length": win_len,
            "anchor_set": win_anchors_name,
            "input_dim": win_input_dim,
            "hidden_size": best_h,
            "num_layers": best_nl,
            "dropout": best_dp,
            "learning_rate": best_lr,
            "batch_size": 512,
            "epochs": 6,
            "lambda_cls": l_cls,
            "seed": 42,
            "notes": f"Stage D5: lambda_cls {l_cls}",
        }
        res_d5 = run_or_load_experiment(cfg_d5, ds_win, win_val_tensor, scaler_win)
        d5_runs.append(res_d5)

    best_lcls = sorted(d5_runs, key=lambda x: x["rmsle_blend_cb"])[0]["lambda_cls"]
    cur_hparams["lambda_cls"] = best_lcls
    print(f"[+] D5 Winner: lambda_cls = {best_lcls}")

    # =========================================================================
    # STAGE E: SEED VERIFICATION (42, 43, 44) & 4-BACKTEST EVALUATION
    # =========================================================================
    print("\n===================================================================")
    print("=== [STAGE E] SEED STABILITY (42, 43, 44) & 4-TEMPORAL BACKTESTS ===")
    print("===================================================================")

    final_hparams = {
        "sequence_length": win_len,
        "anchor_set": win_anchors_name,
        "input_dim": win_input_dim,
        "hidden_size": best_h,
        "num_layers": best_nl,
        "dropout": best_dp,
        "learning_rate": best_lr,
        "lambda_cls": best_lcls,
    }

    # 1. Multi-seed verification (42, 43, 44)
    seed_metrics = []
    for s in [42, 43, 44]:
        cfg_seed = {
            "run_id": f"gru_final_seed{s}",
            **final_hparams,
            "batch_size": 512,
            "epochs": 8,
            "seed": s,
            "notes": f"Stage E: Seed verification seed={s}",
        }
        res_s = run_or_load_experiment(cfg_seed, ds_win, win_val_tensor, scaler_win)
        seed_metrics.append(res_s)

    blend_scores = [r["rmsle_blend_cb"] for r in seed_metrics]
    fact_scores = [r["rmsle_factorized"] for r in seed_metrics]
    mean_blend = float(np.mean(blend_scores))
    std_blend = float(np.std(blend_scores))
    mean_fact = float(np.mean(fact_scores))
    std_fact = float(np.std(fact_scores))

    print(f"\n[*] Multi-Seed Results across 3 Seeds (42, 43, 44):")
    print(f"    - Factorized RMSLE: {mean_fact:.5f} +/- {std_fact:.5f}")
    print(f"    - Blend RMSLE:      {mean_blend:.5f} +/- {std_blend:.5f}")

    # 2. 4-Temporal Backtests Evaluation
    backtest_anchors = [
        date(2025, 10, 13),  # Autumn baseline
        date(2025, 12, 8),   # Pre-New Year surge
        date(2025, 12, 22),  # New Year transition
        date(2026, 1, 14),   # Post-New Year baseline
    ]

    print("\n[*] Evaluating on 4 Fixed Temporal Backtests...")
    backtest_records = []
    for bt_a in backtest_anchors:
        # Load backtest snapshot & target
        y_bt = extract_anchor_targets(data, train_sample_users, bt_a)
        snap_bt = pl.read_parquet(f"data/snapshots/snapshot_{bt_a.strftime('%Y-%m-%d')}.parquet")
        past_gmv_bt = snap_bt["gmv_sum_30d"].to_numpy().astype(np.float32)
        past_buyer_bt = (past_gmv_bt > 0).astype(np.int32)
        del snap_bt

        # Training anchors for this backtest (strictly purged: target window <= bt_a)
        valid_train_anchors = [a for a in win_anchors if a + timedelta(days=30) <= bt_a]
        if len(valid_train_anchors) < 4:
            valid_train_anchors = win_anchors[:6]

        ds_bt, scaler_bt = get_dataset_for_anchors(valid_train_anchors, seq_len=win_len, include_obs=(win_input_dim == 16))
        bt_val_tensor = extract_or_load_master_sequence_tensor(data, train_sample_users, bt_a, max_seq_len=180, include_is_observed=(win_input_dim == 16))

        cfg_bt = {
            "run_id": f"gru_backtest_{bt_a.strftime('%Y%m%d')}",
            **final_hparams,
            "anchor_set": f"purged_for_{bt_a.strftime('%Y%m%d')}",
            "n_anchors": len(valid_train_anchors),
            "batch_size": 512,
            "epochs": 6,
            "seed": 42,
            "notes": f"Stage E: Purged backtest on {bt_a}",
        }
        res_bt = run_gru_training_and_eval(
            cfg_bt, ds_bt, bt_val_tensor, y_bt, past_buyer_bt, train_sample_users, np.zeros_like(y_bt), scaler_bt, device
        )
        backtest_records.append({
            "anchor_date": bt_a.strftime("%Y-%m-%d"),
            "rmsle_factorized": res_bt["rmsle_factorized"],
            "reactivation_auc": res_bt["reactivation_auc"],
            "churn_auc": res_bt["churn_auc"],
            "mse_00": res_bt["mse_00_sleep"],
            "mse_01": res_bt["mse_01_react"],
            "mse_10": res_bt["mse_10_churn"],
            "mse_11": res_bt["mse_11_retention"],
        })

    # Save Backtest Summary
    df_bt_summary = pl.DataFrame(backtest_records)
    df_bt_summary.write_csv(ARTIFACTS_ROOT / "four_backtests_summary.csv")

    total_elapsed = time.time() - t_sweep_start
    print("\n===================================================================")
    print(f"=== FULL MULTITASK GRU SWEEP COMPLETED in {total_elapsed/60:.2f} min! ===")
    print("===================================================================")
    print(f"[*] Final Optimal Configuration: L = {win_len}d | Anchors = {win_anchors_name} | Hidden = {best_h} | Layers = {best_nl} | Drop = {best_dp} | LR = {best_lr} | Lcls = {best_lcls}")
    print(f"[*] Multi-Seed Blend Score: {mean_blend:.5f} +/- {std_blend:.5f} (vs Baseline v5.1: {baseline_blend:.5f}, Delta: {mean_blend - baseline_blend:+.5f})")
    print(f"[*] 4-Backtest Summary:\n{df_bt_summary}")


if __name__ == "__main__":
    main()
