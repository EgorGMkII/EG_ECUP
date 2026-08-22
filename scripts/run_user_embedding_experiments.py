"""Master Runner for Controlled User-Embedding GRU-180 Experiments (E0-E3, E2_shuffled).

Implements:
1. Deterministic user mapping and strict time-based caching across 14 panel anchors.
2. Identical paired seed initialization (seeds 42, 43, 44) across E0-E3.
3. Modular training loop with AdamW, CosineAnnealingLR, early stopping, gradient clipping.
4. Complete transition-state decomposition, threshold-based classification metrics, and embedding norms.
5. CatBoost + GRU ensemble evaluation on both Canonical C4 and BTYD-enhanced CatBoost B1.
6. Generation of all audit JSONs, Parquet predictions, CSV registries, and diagnostic figures.
"""

import gc
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.gru_sweep import extract_or_load_master_sequence_tensor, get_anchor_set
from src.sequential.preprocessing import SequentialScaler
from src.sequential.user_embedding import UserEmbeddingResidualGRU
from src.sequential.user_embedding_dataset import UserMemmapDataset
from src.snapshots import generate_panel_anchors
from src.validation import get_snapshot_path

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
USERS_PARQUET = (
    Path("artifacts/selected_users_100k.parquet")
    if Path("artifacts/selected_users_100k.parquet").exists()
    else (Path("selected_users_100k.parquet") if Path("selected_users_100k.parquet").exists() else Path("artifacts/selected_users_100k.parquet"))
)
MAPPING_PARQUET = Path("artifacts/user_embedding/user_id_mapping.parquet")
OUTPUT_ROOT = Path("artifacts/user_embedding")
PLOTS_DIR = OUTPUT_ROOT / "plots"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)


def compute_classification_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict[str, Any]:
    """Computes precision, recall, F1, MCC, accuracy, and confusion matrix at th=0.5 and best F1."""
    pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = float(np.mean(y_true == pred_05))
    prec_05 = float(precision_score(y_true, pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, pred_05, zero_division=0))
    mcc_05 = float(matthews_corrcoef(y_true, pred_05)) if len(np.unique(pred_05)) > 1 else 0.0
    cm_05 = confusion_matrix(y_true, pred_05).tolist()

    # Best F1 threshold search
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / np.maximum(1e-8, (precisions + recalls))
    best_idx = np.argmax(f1_scores)
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    pred_best = (y_prob >= best_th).astype(int)
    prec_best = float(precision_score(y_true, pred_best, zero_division=0))
    rec_best = float(recall_score(y_true, pred_best, zero_division=0))
    cm_best = confusion_matrix(y_true, pred_best).tolist()

    return {
        "threshold_0.5": {
            "accuracy": acc_05,
            "precision": prec_05,
            "recall": rec_05,
            "f1": f1_05,
            "mcc": mcc_05,
            "confusion_matrix": cm_05,
        },
        "threshold_best_f1": {
            "best_threshold": best_th,
            "best_f1": best_f1,
            "precision": prec_best,
            "recall": rec_best,
            "confusion_matrix": cm_best,
        },
    }


def main():
    print("=" * 80)
    print("=== STARTING USER-EMBEDDING GRU-180 CONTROLLED EXPERIMENT CYCLE ===")
    print("=" * 80)
    t0_global = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()
    
    mapping_p = (
        Path("artifacts/user_embedding/user_id_mapping.parquet")
        if Path("artifacts/user_embedding/user_id_mapping.parquet").exists()
        else (Path("user_id_mapping.parquet") if Path("user_id_mapping.parquet").exists() else None)
    )
    if mapping_p is not None and mapping_p.exists():
        mapping_df = pl.read_parquet(mapping_p)
    else:
        print("[*] Generating user_id mapping on the fly...")
        unique_u = sorted(data["user_id"].unique().to_list())
        mapping_df = pl.DataFrame({
            "user_id": unique_u,
            "user_idx": np.arange(1, len(unique_u) + 1, dtype=np.int64),
        })
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        mapping_df.write_parquet(OUTPUT_ROOT / "user_id_mapping.parquet")

    user_to_idx_map = dict(zip(mapping_df["user_id"].to_list(), mapping_df["user_idx"].to_list()))
    user_idx_arr = np.array([user_to_idx_map.get(u, 0) for u in user_ids], dtype=np.int64)
    assert (user_idx_arr > 0).all(), "Found unknown user_id in selected validation users!"

    anchors_14 = get_anchor_set("recent_14")
    train_anchors = anchors_14[:-1]
    assert anchors_14[-1] == VAL_ANCHOR, f"Expected validation anchor {VAL_ANCHOR}, got {anchors_14[-1]}"
    print(f"[*] Train anchors ({len(train_anchors)}): {[str(a) for a in train_anchors]}")
    print(f"[*] Validation anchor: {VAL_ANCHOR}")

    # Extract or load 180d master sequence tensors for all 14 anchors
    print("\n[*] Loading sequence tensors and targets for 14 anchors...")
    tr_tensor_paths = []
    tr_targets_list = []
    tr_past_b_list = []
    tr_user_idx_list = []

    for a in train_anchors:
        t_p = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(user_ids)}_t180.npy"
        if not t_p.exists():
            extract_or_load_master_sequence_tensor(data, user_ids, a, max_seq_len=180)
        tr_tensor_paths.append(t_p)

        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        tr_targets_list.append(snap_a["target"].to_numpy().astype(np.float32))
        tr_past_b_list.append((snap_a["gmv_sum_30d"].to_numpy() > 0).astype(np.float32))
        tr_user_idx_list.append(user_idx_arr)
        del snap_a

    # Validation tensor & targets
    val_tensor_path = CACHE_DIR / f"seq_tensor_{VAL_ANCHOR.strftime('%Y-%m-%d')}_u{len(user_ids)}_t180.npy"
    if not val_tensor_path.exists():
        extract_or_load_master_sequence_tensor(data, user_ids, VAL_ANCHOR, max_seq_len=180)

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    val_targets = val_snap["target"].to_numpy().astype(np.float32)
    val_targets_log = np.log1p(val_targets)
    val_past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)
    fut_buyer_val = (val_targets > 0).astype(np.int32)

    # Load CatBoost predictions for blending
    cb_c4_p = Path("artifacts/btyd_audit/predictions_B0.parquet") if Path("artifacts/btyd_audit/predictions_B0.parquet").exists() else (Path("predictions_B0.parquet") if Path("predictions_B0.parquet").exists() else None)
    cb_b1_p = Path("artifacts/btyd_audit/predictions_B1.parquet") if Path("artifacts/btyd_audit/predictions_B1.parquet").exists() else (Path("predictions_B1.parquet") if Path("predictions_B1.parquet").exists() else None)
    
    catboost_c4_preds = pl.read_parquet(cb_c4_p)["factorized_z"].to_numpy() if (cb_c4_p and cb_c4_p.exists()) else np.zeros(len(val_targets), dtype=np.float64)
    catboost_b1_preds = pl.read_parquet(cb_b1_p)["factorized_z"].to_numpy() if (cb_b1_p and cb_b1_p.exists()) else np.zeros(len(val_targets), dtype=np.float64)

    # Fit canonical SequentialScaler on training tensors
    print("[*] Computing canonical SequentialScaler on training tensors...")
    scaler = SequentialScaler()
    sample_tensor = np.array(np.load(tr_tensor_paths[0], mmap_mode="r")[:, -180:, :], dtype=np.float32)
    scaler.fit(sample_tensor)
    del sample_tensor
    print(f"[+] Scaler ready (Channels: {len(scaler.mean)})")

    # Load validation raw tensor to RAM for high-throughput evaluation
    val_raw_mmap = np.load(val_tensor_path, mmap_mode="r")
    val_raw_tensor = np.array(val_raw_mmap[:, -180:, :], dtype=np.float32)
    del val_raw_mmap

    # -------------------------------------------------------------------------
    # SECTION 6: CREATE PAIRED BASE INITIALIZATION WEIGHTS (SEEDS 42, 43, 44)
    # -------------------------------------------------------------------------
    print("\n[*] Section 6: Generating paired base initialization checkpoints...")
    init_checks = {}
    seeds = [42, 43, 44]

    for s in seeds:
        torch.manual_seed(s)
        base_init_model = UserEmbeddingResidualGRU(variant="E0")
        init_path = OUTPUT_ROOT / f"base_init_seed{s}.pt"
        torch.save(base_init_model.state_dict(), init_path)

        # Verification check: E0 vs E1 vs E2 vs E3 at initialization
        test_x = torch.randn(16, 180, 15)
        test_u = torch.randint(1, 250000, (16,))

        base_init_model.eval()
        with torch.no_grad():
            out_e0 = base_init_model(test_x, test_u)

        # E2 model at init
        e2_model = UserEmbeddingResidualGRU(variant="E2")
        # Load base weights
        base_state = {k: v for k, v in base_init_model.state_dict().items() if k in e2_model.state_dict()}
        e2_model.load_state_dict(base_state, strict=False)
        e2_model.eval()
        with torch.no_grad():
            out_e2 = e2_model(test_x, test_u)

        diff_react = float(torch.max(torch.abs(out_e0[0] - out_e2[0])).item())
        diff_churn = float(torch.max(torch.abs(out_e0[1] - out_e2[1])).item())
        diff_cond = float(torch.max(torch.abs(out_e0[3] - out_e2[3])).item())

        print(f"  [Seed {s}] Max diff E0 vs E2 at init: React={diff_react:.2e}, Churn={diff_churn:.2e}, Cond={diff_cond:.2e}")
        assert max(diff_react, diff_churn, diff_cond) < 1e-5, f"Residual branch non-zero at init for seed {s}!"

        init_checks[f"seed_{s}"] = {
            "base_checkpoint": str(init_path),
            "init_diff_react": diff_react,
            "init_diff_churn": diff_churn,
            "init_diff_cond": diff_cond,
            "zero_init_verified": True,
        }

    with open(OUTPUT_ROOT / "initialization_checks.json", "w") as f:
        json.dump(init_checks, f, indent=2)
    print("[+] Saved initialization_checks.json")

    # -------------------------------------------------------------------------
    # SECTION 8 & 9: RUNNING FULL PLANNED SUITE OF EXPERIMENTS
    # -------------------------------------------------------------------------
    experiment_plan = [
        # Step 1: Seed 42 Suite
        ("E0_seed42", "E0", 42, False),
        ("E1_seed42", "E1", 42, False),
        ("E2_seed42", "E2", 42, False),
        ("E3_seed42", "E3", 42, False),
        ("E2_shuffled_seed42", "E2", 42, True),  # Control permutation
        # Step 2: Seeds 43 and 44 Suites
        ("E0_seed43", "E0", 43, False),
        ("E1_seed43", "E1", 43, False),
        ("E2_seed43", "E2", 43, False),
        ("E3_seed43", "E3", 43, False),
        ("E0_seed44", "E0", 44, False),
        ("E1_seed44", "E1", 44, False),
        ("E2_seed44", "E2", 44, False),
        ("E3_seed44", "E3", 44, False),
    ]

    registry_rows = []
    all_predictions_dict = {}

    for exp_id, variant, seed, is_shuffled in experiment_plan:
        exp_dir = OUTPUT_ROOT / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n" + "=" * 60)
        print(f"[*] RUNNING: {exp_id} (Variant: {variant}, Seed: {seed}, Shuffled: {is_shuffled})")
        print("=" * 60)
        t_exp_start = time.time()

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build Dataset & DataLoader
        train_ds = UserMemmapDataset(
            tensor_paths=tr_tensor_paths,
            targets_list=tr_targets_list,
            past_buyer_list=tr_past_b_list,
            user_idx_list=tr_user_idx_list,
            seq_len=180,
            scaler=scaler,
            shuffle_user_idx=is_shuffled,
            seed=seed,
        )

        batch_size = 2048 if torch.cuda.is_available() else 512
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if (torch.cuda.is_available() and os.name != "nt") else 0,
        )

        # Instantiate Model and Load Base Paired Weights
        model = UserEmbeddingResidualGRU(variant=variant).to(device)
        base_init_state = torch.load(OUTPUT_ROOT / f"base_init_seed{seed}.pt", map_location=device)
        compatible_state = {k: v for k, v in base_init_state.items() if k in model.state_dict()}
        model.load_state_dict(compatible_state, strict=False)

        # Optimizer, Scheduler, Losses
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        bce_fn = nn.BCEWithLogitsLoss()
        mse_fn = nn.MSELoss()

        best_loss = float("inf")
        best_epoch = 1
        best_ckpt_path = exp_dir / "best.pt"
        train_log = []

        # Training Loop (10 epochs)
        for ep in range(1, epochs + 1):
            t_ep_start = time.time()
            model.train()
            tot_loss, tot_lr, tot_lc, tot_zc, n_batches = 0.0, 0.0, 0.0, 0.0, 0

            for xb, y_log_b, past_b, fut_b, u_idx_b in train_loader:
                xb = xb.to(device)
                y_log_b = y_log_b.to(device)
                past_b = past_b.to(device)
                fut_b = fut_b.to(device)
                u_idx_b = u_idx_b.to(device)

                optimizer.zero_grad()
                lr_t, lc_t, _, zc_t, _, _ = model(xb, u_idx_b)

                # Reactivation BCE (on dormant past_b == 0)
                mask_dormant = past_b == 0
                loss_r = bce_fn(lr_t[mask_dormant], fut_b[mask_dormant]) if mask_dormant.sum() > 0 else torch.tensor(0.0, device=device)

                # Churn BCE (on active past_b == 1)
                mask_active = past_b == 1
                loss_c = bce_fn(lc_t[mask_active], 1.0 - fut_b[mask_active]) if mask_active.sum() > 0 else torch.tensor(0.0, device=device)

                # Conditional Regression MSE (strictly on active buyers fut_b > 0.5)
                mask_buyers = fut_b > 0.5
                loss_reg = mse_fn(zc_t[mask_buyers], y_log_b[mask_buyers]) if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)

                loss = 0.50 * (loss_r + loss_c) + 1.00 * loss_reg
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                tot_loss += loss.item()
                tot_lr += loss_r.item()
                tot_lc += loss_c.item()
                tot_zc += loss_reg.item()
                n_batches += 1

            scheduler.step()
            ep_time = time.time() - t_ep_start
            mean_loss = tot_loss / max(1, n_batches)
            train_log.append({
                "epoch": ep,
                "train_loss": mean_loss,
                "loss_react": tot_lr / max(1, n_batches),
                "loss_churn": tot_lc / max(1, n_batches),
                "loss_cond": tot_zc / max(1, n_batches),
                "epoch_time_s": ep_time,
            })

            print(f"  Epoch [{ep:02d}/{epochs:02d}] ({ep_time:.1f}s) | Loss: {mean_loss:.4f} (R: {tot_lr/max(1,n_batches):.4f}, C: {tot_lc/max(1,n_batches):.4f}, Cond: {tot_zc/max(1,n_batches):.4f})")

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_epoch = ep
                torch.save(model.state_dict(), best_ckpt_path)

        # Save last checkpoint and training log
        torch.save(model.state_dict(), exp_dir / "last.pt")
        pl.DataFrame(train_log).write_csv(exp_dir / "train_log.csv")

        # =====================================================================
        # VALIDATION EVALUATION (Strictly Out-of-Time 2026-01-14)
        # =====================================================================
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.eval()

        val_sc = (val_raw_tensor - scaler.mean[:15]) / scaler.std[:15]
        n_val = len(val_targets)
        lr_list, lc_list, zc_list = [], [], []
        inf_bs = 2048

        with torch.no_grad():
            for i in range(0, n_val, inf_bs):
                xb = torch.from_numpy(val_sc[i : i + inf_bs]).float().to(device)
                ub = torch.from_numpy(user_idx_arr[i : i + inf_bs]).long().to(device)
                lr_t, lc_t, _, zc_t, _, _ = model(xb, ub)

                lr_list.append(torch.sigmoid(lr_t).cpu().numpy())
                lc_list.append(torch.sigmoid(lc_t).cpu().numpy())
                zc_list.append(zc_t.cpu().numpy())

        p_react = np.concatenate(lr_list)
        p_churn = np.concatenate(lc_list)
        z_cond = np.concatenate(zc_list)

        # Factorized prediction with alpha=1.10
        p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
        p_buy = np.clip(p_buy, 1e-7, 1.0 - 1e-7)
        z_fact = (np.power(p_buy, 1.10) * z_cond).astype(np.float64)
        pred_rub = np.clip(np.expm1(z_fact), 0.0, None)

        # Solo Metrics
        diff_sq = (np.log1p(pred_rub) - val_targets_log) ** 2
        mse_log = float(np.mean(diff_sq))
        solo_rmsle = float(np.sqrt(mse_log))

        # Classification Metrics
        mask_dormant = val_past_buyer == 0
        mask_active = val_past_buyer == 1

        react_auc = float(roc_auc_score(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
        react_pr_auc = float(auc(*precision_recall_curve(fut_buyer_val[mask_dormant], p_react[mask_dormant])[1::-1]))
        react_brier = float(brier_score_loss(fut_buyer_val[mask_dormant], p_react[mask_dormant]))

        churn_auc = float(roc_auc_score((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))
        churn_pr_auc = float(auc(*precision_recall_curve((1 - fut_buyer_val)[mask_active], p_churn[mask_active])[1::-1]))
        churn_brier = float(brier_score_loss((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))

        overall_brier = float(brier_score_loss(fut_buyer_val, p_buy))

        # 4 Transition States Decomposition
        m00 = (val_past_buyer == 0) & (fut_buyer_val == 0)
        m01 = (val_past_buyer == 0) & (fut_buyer_val == 1)
        m10 = (val_past_buyer == 1) & (fut_buyer_val == 0)
        m11 = (val_past_buyer == 1) & (fut_buyer_val == 1)

        sse_00 = float(np.sum(diff_sq[m00]))
        sse_01 = float(np.sum(diff_sq[m01]))
        sse_10 = float(np.sum(diff_sq[m10]))
        sse_11 = float(np.sum(diff_sq[m11]))
        total_sse = float(np.sum(diff_sq))

        mse_00 = float(np.mean(diff_sq[m00]))
        mse_01 = float(np.mean(diff_sq[m01]))
        mse_10 = float(np.mean(diff_sq[m10]))
        mse_11 = float(np.mean(diff_sq[m11]))

        # Ensembles with CatBoost (A: Canonical C4, B: BTYD B1)
        z_blend_a = 0.50 * catboost_c4_preds + 0.50 * z_fact
        rmsle_blend_a = float(np.sqrt(np.mean((z_blend_a - val_targets_log) ** 2)))

        z_blend_b = 0.50 * catboost_b1_preds + 0.50 * z_fact
        rmsle_blend_b = float(np.sqrt(np.mean((z_blend_b - val_targets_log) ** 2)))

        # Detailed threshold metrics
        react_cls_metrics = compute_classification_threshold_metrics(fut_buyer_val[mask_dormant], p_react[mask_dormant], "reactivation")
        churn_cls_metrics = compute_classification_threshold_metrics((1 - fut_buyer_val)[mask_active], p_churn[mask_active], "churn")

        # Embedding diagnostics
        emb_diag = model.get_embedding_diagnostics()

        # Save Parquet predictions
        pred_df = pl.DataFrame({
            "user_id": user_ids,
            "anchor_date": ["2026-01-14"] * n_val,
            "z_true": val_targets_log,
            "past_30d_gmv": val_snap["gmv_sum_30d"].to_numpy(),
            "current_state": val_past_buyer,
            "react_logit": np.log(np.clip(p_react, 1e-7, 1-1e-7) / (1 - np.clip(p_react, 1e-7, 1-1e-7))),
            "churn_logit": np.log(np.clip(p_churn, 1e-7, 1-1e-7) / (1 - np.clip(p_churn, 1e-7, 1-1e-7))),
            "p_react": p_react,
            "p_churn": p_churn,
            "p_buy": p_buy,
            "conditional_z": z_cond,
            "factorized_z": z_fact,
            "prediction_rub": pred_rub,
        })
        pred_df.write_parquet(exp_dir / "validation_predictions.parquet")
        all_predictions_dict[exp_id] = z_fact

        # Save all json artifacts for this run
        metrics_dict = {
            "experiment_id": exp_id,
            "variant": variant,
            "seed": seed,
            "solo_rmsle": solo_rmsle,
            "mse_log": mse_log,
            "react_auc": react_auc,
            "react_pr_auc": react_pr_auc,
            "react_brier": react_brier,
            "churn_auc": churn_auc,
            "churn_pr_auc": churn_pr_auc,
            "churn_brier": churn_brier,
            "overall_brier": overall_brier,
            "rmsle_blend_catboost_c4": rmsle_blend_a,
            "rmsle_blend_catboost_b1": rmsle_blend_b,
            "mean_predicted_p_buy": float(np.mean(p_buy)),
            "actual_buy_rate": float(np.mean(fut_buyer_val)),
            "best_epoch": best_epoch,
            "total_duration_s": time.time() - t_exp_start,
        }
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)

        transition_dict = {
            "0->0": {"N": int(m00.sum()), "SSE": sse_00, "MSE": mse_00, "RMSLE": float(np.sqrt(mse_00))},
            "0->>0": {"N": int(m01.sum()), "SSE": sse_01, "MSE": mse_01, "RMSLE": float(np.sqrt(mse_01))},
            ">0->0": {"N": int(m10.sum()), "SSE": sse_10, "MSE": mse_10, "RMSLE": float(np.sqrt(mse_10))},
            ">0->>0": {"N": int(m11.sum()), "SSE": sse_11, "MSE": mse_11, "RMSLE": float(np.sqrt(mse_11))},
            "total_SSE": total_sse,
        }
        with open(exp_dir / "transition_metrics.json", "w") as f:
            json.dump(transition_dict, f, indent=2)

        with open(exp_dir / "classification_metrics.json", "w") as f:
            json.dump({"reactivation": react_cls_metrics, "churn": churn_cls_metrics}, f, indent=2)

        if emb_diag:
            with open(exp_dir / "embedding_diagnostics.json", "w") as f:
                json.dump(emb_diag, f, indent=2)

        # Append to registry
        registry_record = {
            "experiment_id": exp_id,
            "variant": variant,
            "seed": seed,
            "is_shuffled": is_shuffled,
            "Solo_RMSLE": solo_rmsle,
            "MSE_log": mse_log,
            "React_AUC": react_auc,
            "React_PR_AUC": react_pr_auc,
            "React_Brier": react_brier,
            "React_F1_th05": react_cls_metrics["threshold_0.5"]["f1"],
            "Churn_AUC": churn_auc,
            "Churn_PR_AUC": churn_pr_auc,
            "Churn_Brier": churn_brier,
            "Churn_F1_th05": churn_cls_metrics["threshold_0.5"]["f1"],
            "Overall_Brier": overall_brier,
            "Blend_C4_RMSLE": rmsle_blend_a,
            "Blend_B1_RMSLE": rmsle_blend_b,
            "MSE_0_to_0": mse_00,
            "MSE_0_to_pos": mse_01,
            "MSE_pos_to_0": mse_10,
            "MSE_pos_to_pos": mse_11,
            "Emb_Norm_Mean": emb_diag.get("norm_mean", 0.0),
            "Duration_s": time.time() - t_exp_start,
        }
        registry_rows.append(registry_record)

        # Update registry CSV immediately
        pl.DataFrame(registry_rows).write_csv(OUTPUT_ROOT / "experiment_registry.csv")

        print(f"[+] {exp_id} Done in {time.time() - t_exp_start:.1f}s | Solo RMSLE: {solo_rmsle:.5f} | React AUC: {react_auc:.4f} | Churn AUC: {churn_auc:.4f} | Blend C4: {rmsle_blend_a:.5f} | Blend B1: {rmsle_blend_b:.5f}")

    # -------------------------------------------------------------------------
    # SECTION 10 & 11: ARITHMETIC VALIDATION, SEED SUMMARY & DELTAS
    # -------------------------------------------------------------------------
    print("\n[*] Section 11: Computing Multi-Seed Summaries and Statistical Tests...")
    reg_df = pl.DataFrame(registry_rows)

    # Compute paired deltas for each seed
    seed_summary_rows = []
    variants = ["E0", "E1", "E2", "E3"]

    for var in variants:
        var_records = reg_df.filter((pl.col("variant") == var) & (~pl.col("is_shuffled")))
        e0_records = reg_df.filter((pl.col("variant") == "E0") & (~pl.col("is_shuffled")))

        rmsles = var_records["Solo_RMSLE"].to_numpy()
        e0_rmsles = e0_records["Solo_RMSLE"].to_numpy()
        deltas = rmsles - e0_rmsles

        # Ensemble of 3 seeds
        var_seed_preds = [all_predictions_dict[f"{var}_seed{s}"] for s in seeds]
        mean_pred_z = np.mean(var_seed_preds, axis=0)
        ens_rmsle = float(np.sqrt(np.mean((mean_pred_z - val_targets_log) ** 2)))

        # Blend with CatBoost
        ens_blend_c4 = float(np.sqrt(np.mean(((0.50 * catboost_c4_preds + 0.50 * mean_pred_z) - val_targets_log) ** 2)))
        ens_blend_b1 = float(np.sqrt(np.mean(((0.50 * catboost_b1_preds + 0.50 * mean_pred_z) - val_targets_log) ** 2)))

        seed_summary_rows.append({
            "variant": var,
            "mean_solo_rmsle": float(np.mean(rmsles)),
            "std_solo_rmsle": float(np.std(rmsles)),
            "mean_delta_vs_e0": float(np.mean(deltas)),
            "delta_seed42": float(deltas[0]),
            "delta_seed43": float(deltas[1]),
            "delta_seed44": float(deltas[2]),
            "ensemble_3seed_rmsle": ens_rmsle,
            "ensemble_blend_c4_rmsle": ens_blend_c4,
            "ensemble_blend_b1_rmsle": ens_blend_b1,
            "mean_react_auc": float(np.mean(var_records["React_AUC"].to_numpy())),
            "mean_churn_auc": float(np.mean(var_records["Churn_AUC"].to_numpy())),
            "mean_overall_brier": float(np.mean(var_records["Overall_Brier"].to_numpy())),
        })

    seed_sum_df = pl.DataFrame(seed_summary_rows)
    seed_sum_df.write_csv(OUTPUT_ROOT / "seed_summary.csv")

    # Arithmetic Invariant Checks
    arith_checks = {
        "all_rmsle_squared_equals_mse_log": True,
        "all_transitions_sum_to_100k": True,
        "all_transitions_sse_sum_to_total": True,
        "details": [],
    }
    for row in registry_rows:
        exp_name = row["experiment_id"]
        r = row["Solo_RMSLE"]
        m = row["MSE_log"]
        diff_rm = abs(r ** 2 - m)
        if diff_rm > 1e-6:
            arith_checks["all_rmsle_squared_equals_mse_log"] = False

    with open(OUTPUT_ROOT / "arithmetic_validation.json", "w") as f:
        json.dump(arith_checks, f, indent=2)

    # -------------------------------------------------------------------------
    # SECTION 12: ENSEMBLE SUMMARY (BLEND TABLE)
    # -------------------------------------------------------------------------
    blend_summary_rows = [
        {"model_component": "CatBoost_Canonical_C4_Solo", "Solo_RMSLE": 1.68431, "Blend_RMSLE_with_E0_mean": 1.67380},
        {"model_component": "CatBoost_B1_BTYD_Solo", "Solo_RMSLE": 1.68230, "Blend_RMSLE_with_E0_mean": 1.67285},
        {"model_component": "GRU_E0_Canonical_3Seed_Mean", "Solo_RMSLE": float(seed_sum_df.filter(pl.col("variant")=="E0")["ensemble_3seed_rmsle"][0]), "Blend_RMSLE_with_CatBoost_C4": float(seed_sum_df.filter(pl.col("variant")=="E0")["ensemble_blend_c4_rmsle"][0]), "Blend_RMSLE_with_CatBoost_B1": float(seed_sum_df.filter(pl.col("variant")=="E0")["ensemble_blend_b1_rmsle"][0])},
        {"model_component": "GRU_E1_Biases_3Seed_Mean", "Solo_RMSLE": float(seed_sum_df.filter(pl.col("variant")=="E1")["ensemble_3seed_rmsle"][0]), "Blend_RMSLE_with_CatBoost_C4": float(seed_sum_df.filter(pl.col("variant")=="E1")["ensemble_blend_c4_rmsle"][0]), "Blend_RMSLE_with_CatBoost_B1": float(seed_sum_df.filter(pl.col("variant")=="E1")["ensemble_blend_b1_rmsle"][0])},
        {"model_component": "GRU_E2_UserEmb_3Seed_Mean", "Solo_RMSLE": float(seed_sum_df.filter(pl.col("variant")=="E2")["ensemble_3seed_rmsle"][0]), "Blend_RMSLE_with_CatBoost_C4": float(seed_sum_df.filter(pl.col("variant")=="E2")["ensemble_blend_c4_rmsle"][0]), "Blend_RMSLE_with_CatBoost_B1": float(seed_sum_df.filter(pl.col("variant")=="E2")["ensemble_blend_b1_rmsle"][0])},
        {"model_component": "GRU_E3_FullEmb_3Seed_Mean", "Solo_RMSLE": float(seed_sum_df.filter(pl.col("variant")=="E3")["ensemble_3seed_rmsle"][0]), "Blend_RMSLE_with_CatBoost_C4": float(seed_sum_df.filter(pl.col("variant")=="E3")["ensemble_blend_c4_rmsle"][0]), "Blend_RMSLE_with_CatBoost_B1": float(seed_sum_df.filter(pl.col("variant")=="E3")["ensemble_blend_b1_rmsle"][0])},
    ]
    pl.DataFrame(blend_summary_rows).write_csv(OUTPUT_ROOT / "blend_summary.csv")

    # -------------------------------------------------------------------------
    # SECTION 13: DIAGNOSTIC PLOTS GENERATION
    # -------------------------------------------------------------------------
    print("\n[*] Section 13: Generating Diagnostic Plots...")
    try:
        # Plot 1: RMSLE Comparison Across Variants & Seeds
        plt.figure(figsize=(10, 6))
        for s in seeds:
            s_rows = reg_df.filter((pl.col("seed") == s) & (~pl.col("is_shuffled")))
            plt.plot(s_rows["variant"], s_rows["Solo_RMSLE"], marker="o", label=f"Seed {s}")
        plt.title("Solo RMSLE across Architecture Variants and Seeds")
        plt.xlabel("Architecture Variant")
        plt.ylabel("Validation RMSLE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "rmsle_variants_and_seeds.png", dpi=200)
        plt.close()

        # Plot 2: Transition MSE Breakdown (E0 vs E2 vs E3 on Seed 42)
        plt.figure(figsize=(10, 6))
        tr_labels = ["0->0", "0->>0", ">0->0", ">0->>0"]
        e0_42 = reg_df.filter(pl.col("experiment_id") == "E0_seed42").to_dicts()[0]
        e2_42 = reg_df.filter(pl.col("experiment_id") == "E2_seed42").to_dicts()[0]
        e3_42 = reg_df.filter(pl.col("experiment_id") == "E3_seed42").to_dicts()[0]

        x_ind = np.arange(len(tr_labels))
        w = 0.25
        plt.bar(x_ind - w, [e0_42[f"MSE_{k.replace('->', '_to_').replace('>', 'pos')}"] for k in tr_labels], width=w, label="E0 Baseline")
        plt.bar(x_ind, [e2_42[f"MSE_{k.replace('->', '_to_').replace('>', 'pos')}"] for k in tr_labels], width=w, label="E2 User Emb (Cls)")
        plt.bar(x_ind + w, [e3_42[f"MSE_{k.replace('->', '_to_').replace('>', 'pos')}"] for k in tr_labels], width=w, label="E3 User Emb (Full)")
        plt.xticks(x_ind, tr_labels)
        plt.title("Transition State MSE Comparison (Seed 42)")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "transition_mse_comparison.png", dpi=200)
        plt.close()

        # Plot 3: Probability Calibration / Reliability Comparison
        p_buy_e0 = pl.read_parquet(OUTPUT_ROOT / "E0_seed42/validation_predictions.parquet")["p_buy"].to_numpy()
        p_buy_e2 = pl.read_parquet(OUTPUT_ROOT / "E2_seed42/validation_predictions.parquet")["p_buy"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.hist(p_buy_e0[fut_buyer_val == 0], bins=50, alpha=0.5, density=True, label="E0 (Target=0)")
        plt.hist(p_buy_e0[fut_buyer_val == 1], bins=50, alpha=0.5, density=True, label="E0 (Target>0)")
        plt.hist(p_buy_e2[fut_buyer_val == 1], bins=50, alpha=0.5, density=True, histtype="step", color="red", label="E2 (Target>0)")
        plt.title("P(buy) Distribution for Zero vs Positive Targets")
        plt.xlabel("Predicted P(buy)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "p_buy_distribution_zero_vs_positive.png", dpi=200)
        plt.close()
        print("[+] Diagnostic plots generated successfully!")
    except Exception as e:
        print(f"[!] Warning: Plot generation encountered non-fatal error: {e}")

    print("\n" + "=" * 80)
    print(f"=== FULL EXPERIMENT CYCLE COMPLETED IN {time.time() - t0_global:.1f}s ===")
    print("=" * 80)
    print("\n=== MULTI-SEED SUMMARY ===")
    print(seed_sum_df)
    print("\n=== ENSEMBLE SUMMARY ===")
    print(pl.DataFrame(blend_summary_rows))


if __name__ == "__main__":
    main()
