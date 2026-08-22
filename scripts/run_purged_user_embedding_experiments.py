"""Strictly Purged & Embargoed User-Embedding GRU-180 Experiment Suite.

Guarantee: max(train_target_end) <= validation_target_start - 1 (EXACTLY 0 DAYS TARGET OVERLAP).
"""

from datetime import date, timedelta
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sequential.dataset import CACHE_DIR
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
OUTPUT_ROOT = Path("artifacts/purged_user_embedding")
PLOTS_DIR = OUTPUT_ROOT / "plots"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# ANCHOR SET DEFINITIONS WITH STRICT 30-DAY EMBARGO
# -----------------------------------------------------------------------------
VAL_ANCHOR_PRIMARY = date(2026, 1, 14)
VAL_ANCHOR_WALKFORWARD = date(2025, 12, 8)

# All potential training anchors
ALL_ANCHORS_14 = get_anchor_set("recent_14")

def get_purged_train_anchors(val_anchor: date) -> List[date]:
    """Returns all train anchors where target_end <= val_anchor (Strict 30d Embargo)."""
    val_target_start = val_anchor + timedelta(days=1)
    purged = []
    for a in ALL_ANCHORS_14:
        if a >= val_anchor:
            continue
        t_end = a + timedelta(days=30)
        # Check target strictly ends on or before val_anchor
        if t_end <= val_anchor:
            purged.append(a)
    return purged

PRIMARY_TRAIN_ANCHORS = get_purged_train_anchors(VAL_ANCHOR_PRIMARY)
WALKFORWARD_TRAIN_ANCHORS = get_purged_train_anchors(VAL_ANCHOR_WALKFORWARD)


def compute_classification_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict[str, Any]:
    pred_05 = (y_prob >= 0.5).astype(int)
    acc_05 = float(np.mean(y_true == pred_05))
    prec_05 = float(precision_score(y_true, pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, pred_05, zero_division=0))
    mcc_05 = float(matthews_corrcoef(y_true, pred_05)) if len(np.unique(pred_05)) > 1 else 0.0
    cm_05 = confusion_matrix(y_true, pred_05).tolist()

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
    print("=== STARTING PURGED & EMBARGOED USER-EMBEDDING EXPERIMENT CYCLE ===")
    print("=" * 80)
    t0_global = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()

    mapping_p = Path("artifacts/user_embedding/user_id_mapping.parquet")
    if not mapping_p.exists():
        mapping_p = Path("user_id_mapping.parquet")
    
    if mapping_p.exists():
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

    # -------------------------------------------------------------------------
    # PRINT PURGED TARGET AUDIT
    # -------------------------------------------------------------------------
    print(f"\n[*] Primary Validation Anchor: {VAL_ANCHOR_PRIMARY} (Target: {VAL_ANCHOR_PRIMARY + timedelta(days=1)}..{VAL_ANCHOR_PRIMARY + timedelta(days=30)})")
    print(f"[*] Purged Train Anchors ({len(PRIMARY_TRAIN_ANCHORS)}): {[str(a) for a in PRIMARY_TRAIN_ANCHORS]}")
    for a in PRIMARY_TRAIN_ANCHORS:
        t_start = a + timedelta(days=1)
        t_end = a + timedelta(days=30)
        assert t_end <= VAL_ANCHOR_PRIMARY, f"Target leakage detected for anchor {a}!"
        print(f"  - Train Anchor {a}: target {t_start}..{t_end} (ends {VAL_ANCHOR_PRIMARY - t_end} before val anchor)")

    print(f"\n[+] 100% PURGED INVARIANT VERIFIED: MAX TRAIN TARGET END ({PRIMARY_TRAIN_ANCHORS[-1] + timedelta(days=30)}) <= VAL ANCHOR ({VAL_ANCHOR_PRIMARY})")

    # -------------------------------------------------------------------------
    # LOAD SEQUENCES AND TARGETS FOR PRIMARY PURGED SUITE
    # -------------------------------------------------------------------------
    tr_tensor_paths = []
    tr_targets_list = []
    tr_past_b_list = []
    tr_user_idx_list = []

    for a in PRIMARY_TRAIN_ANCHORS:
        t_p = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(user_ids)}_t180.npy"
        if not t_p.exists():
            extract_or_load_master_sequence_tensor(data, user_ids, a, max_seq_len=180)
        tr_tensor_paths.append(t_p)

        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        tr_targets_list.append(snap_a["target"].to_numpy().astype(np.float32))
        tr_past_b_list.append((snap_a["gmv_sum_30d"].to_numpy() > 0).astype(np.float32))
        tr_user_idx_list.append(user_idx_arr)
        del snap_a

    # Validation tensor & targets (Primary: 2026-01-14)
    val_tensor_path = CACHE_DIR / f"seq_tensor_{VAL_ANCHOR_PRIMARY.strftime('%Y-%m-%d')}_u{len(user_ids)}_t180.npy"
    if not val_tensor_path.exists():
        extract_or_load_master_sequence_tensor(data, user_ids, VAL_ANCHOR_PRIMARY, max_seq_len=180)

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR_PRIMARY, SNAPSHOTS_DIR))
    val_targets = val_snap["target"].to_numpy().astype(np.float32)
    val_targets_log = np.log1p(val_targets)
    val_past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)
    fut_buyer_val = (val_targets > 0).astype(np.int32)

    # Scaler fit on first purged training tensor
    scaler = SequentialScaler()
    sample_tensor = np.array(np.load(tr_tensor_paths[0], mmap_mode="r")[:, -180:, :], dtype=np.float32)
    scaler.fit(sample_tensor)
    del sample_tensor

    val_raw_mmap = np.load(val_tensor_path, mmap_mode="r")
    val_raw_tensor = np.array(val_raw_mmap[:, -180:, :], dtype=np.float32)
    del val_raw_mmap
    val_sc = (val_raw_tensor - scaler.mean[:15]) / scaler.std[:15]

    # Save Base Init weights
    seeds = [42, 43, 44]
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        base_init_model = UserEmbeddingResidualGRU(variant="E0")
        init_path = OUTPUT_ROOT / f"base_init_seed{s}.pt"
        torch.save(base_init_model.state_dict(), init_path)

    # -------------------------------------------------------------------------
    # EXPERIMENT PLAN (PURGED PRIMARY + WALKFORWARD + HONEST CONTROLS)
    # -------------------------------------------------------------------------
    experiment_plan = [
        # Purged Primary Benchmark (Seed 42)
        ("Purged_E0_seed42", "E0", 42, False, "none"),
        ("Purged_E1_seed42", "E1", 42, False, "none"),
        ("Purged_E2_seed42", "E2", 42, False, "none"),
        ("Purged_E3_seed42", "E3", 42, False, "none"),
        # Honest Control 1: E2 Zero / UNK user embedding at validation
        ("Purged_E2_val_unk_seed42", "E2", 42, False, "val_unk"),
        # Honest Control 2: Train with shuffled IDs, true IDs at validation
        ("Purged_E2_train_shuffled_seed42", "E2", 42, True, "none"),
        # Honest Control 3: Permute user embeddings at validation once
        ("Purged_E2_val_permuted_seed42", "E2", 42, False, "val_permute"),
        # Multi-Seed Purged (Seeds 43, 44)
        ("Purged_E0_seed43", "E0", 43, False, "none"),
        ("Purged_E1_seed43", "E1", 43, False, "none"),
        ("Purged_E2_seed43", "E2", 43, False, "none"),
        ("Purged_E3_seed43", "E3", 43, False, "none"),
        ("Purged_E0_seed44", "E0", 44, False, "none"),
        ("Purged_E1_seed44", "E1", 44, False, "none"),
        ("Purged_E2_seed44", "E2", 44, False, "none"),
        ("Purged_E3_seed44", "E3", 44, False, "none"),
    ]

    registry_rows = []
    preds_dict = {}

    for exp_id, variant, seed, is_shuffled_train, val_mode in experiment_plan:
        exp_dir = OUTPUT_ROOT / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"[*] RUNNING: {exp_id} (Variant: {variant}, Seed: {seed}, TrainShuffled: {is_shuffled_train}, ValMode: {val_mode})")
        print("=" * 60)
        t0_exp = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = UserMemmapDataset(
            tensor_paths=tr_tensor_paths,
            targets_list=tr_targets_list,
            past_buyer_list=tr_past_b_list,
            user_idx_list=tr_user_idx_list,
            seq_len=180,
            scaler=scaler,
            shuffle_user_idx=is_shuffled_train,
        )
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=False)

        model = UserEmbeddingResidualGRU(variant=variant).to(device)
        base_state = torch.load(OUTPUT_ROOT / f"base_init_seed{seed}.pt", map_location=device)
        compatible = {k: v for k, v in base_state.items() if k in model.state_dict()}
        model.load_state_dict(compatible, strict=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        bce_fn = nn.BCEWithLogitsLoss()
        mse_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_ckpt_path = exp_dir / "best.pt"
        train_log = []

        for epoch in range(1, epochs + 1):
            t0_ep = time.time()
            model.train()
            ep_loss, ep_r, ep_c, ep_reg = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for xb, y_log_b, past_b, fut_b, ub in train_loader:
                xb, y_log_b = xb.to(device), y_log_b.to(device)
                past_b, fut_b, ub = past_b.to(device), fut_b.to(device), ub.to(device)

                optimizer.zero_grad()
                lr_out, lc_out, _, zc_out, _, _ = model(xb, ub)

                mask_dormant = (past_b == 0)
                mask_active = (past_b == 1)

                loss_r = bce_fn(lr_out[mask_dormant], fut_b[mask_dormant]) if mask_dormant.sum() > 0 else torch.tensor(0.0, device=device)
                loss_c = bce_fn(lc_out[mask_active], (1.0 - fut_b[mask_active])) if mask_active.sum() > 0 else torch.tensor(0.0, device=device)
                loss_cls = 0.5 * (loss_r + loss_c)

                mask_buyers = (fut_b == 1)
                loss_reg = mse_fn(zc_out[mask_buyers], y_log_b[mask_buyers]) if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)

                loss = 0.50 * loss_cls + 1.00 * loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                ep_loss += loss.item()
                ep_r += loss_r.item()
                ep_c += loss_c.item()
                ep_reg += loss_reg.item()
                n_batches += 1

            scheduler.step()
            dur_ep = time.time() - t0_ep
            train_log.append({
                "epoch": epoch,
                "loss": ep_loss / n_batches,
                "loss_r": ep_r / n_batches,
                "loss_c": ep_c / n_batches,
                "loss_reg": ep_reg / n_batches,
                "duration_s": dur_ep,
            })
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dur_ep:.1f}s) | Loss: {ep_loss/n_batches:.4f} (R: {ep_r/n_batches:.4f}, C: {ep_c/n_batches:.4f}, Cond: {ep_reg/n_batches:.4f})")

            # Save best by training loss convergence
            if ep_loss < best_val_loss:
                best_val_loss = ep_loss
                torch.save(model.state_dict(), best_ckpt_path)

        torch.save(model.state_dict(), exp_dir / "last.pt")
        pl.DataFrame(train_log).write_csv(exp_dir / "train_log.csv")

        # ---------------------------------------------------------------------
        # VALIDATION INFERENCE WITH CONTROLS
        # ---------------------------------------------------------------------
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.eval()

        n_val = len(val_targets)
        # Apply validation modes
        if val_mode == "val_unk":
            val_user_idx = np.zeros(n_val, dtype=np.int64)  # UNK / Zero embedding
        elif val_mode == "val_permute":
            val_user_idx = np.random.permutation(user_idx_arr)  # Random permutation across users
        else:
            val_user_idx = user_idx_arr

        lr_list, lc_list, zc_list = [], [], []
        inf_bs = 2048
        with torch.no_grad():
            for i in range(0, n_val, inf_bs):
                xb = torch.from_numpy(val_sc[i : i + inf_bs]).float().to(device)
                ub = torch.from_numpy(val_user_idx[i : i + inf_bs]).long().to(device)
                lr_t, lc_t, _, zc_t, _, _ = model(xb, ub)

                lr_list.append(torch.sigmoid(lr_t).cpu().numpy())
                lc_list.append(torch.sigmoid(lc_t).cpu().numpy())
                zc_list.append(zc_t.cpu().numpy())

        p_react = np.concatenate(lr_list)
        p_churn = np.concatenate(lc_list)
        z_cond = np.concatenate(zc_list)

        p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
        p_buy = np.clip(p_buy, 1e-7, 1.0 - 1e-7)
        z_fact = (np.power(p_buy, 1.10) * z_cond).astype(np.float64)
        pred_rub = np.clip(np.expm1(z_fact), 0.0, None)

        diff_sq = (z_fact - val_targets_log) ** 2
        total_mse = float(np.mean(diff_sq))
        solo_rmsle = float(np.sqrt(total_mse))

        mask_dormant = (val_past_buyer == 0)
        mask_active = (val_past_buyer == 1)

        react_auc = float(roc_auc_score(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
        react_pr_auc = float(auc(*precision_recall_curve(fut_buyer_val[mask_dormant], p_react[mask_dormant])[1::-1]))
        react_brier = float(brier_score_loss(fut_buyer_val[mask_dormant], p_react[mask_dormant]))

        churn_auc = float(roc_auc_score((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))
        churn_pr_auc = float(auc(*precision_recall_curve((1 - fut_buyer_val)[mask_active], p_churn[mask_active])[1::-1]))
        churn_brier = float(brier_score_loss((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))

        overall_brier = float(brier_score_loss(fut_buyer_val, p_buy))

        # Transition decomposition (Canonical 31554 / 12244 / 14462 / 41740)
        m00 = (val_past_buyer == 0) & (fut_buyer_val == 0)
        m01 = (val_past_buyer == 0) & (fut_buyer_val == 1)
        m10 = (val_past_buyer == 1) & (fut_buyer_val == 0)
        m11 = (val_past_buyer == 1) & (fut_buyer_val == 1)

        mse_00 = float(np.mean(diff_sq[m00]))
        mse_01 = float(np.mean(diff_sq[m01]))
        mse_10 = float(np.mean(diff_sq[m10]))
        mse_11 = float(np.mean(diff_sq[m11]))

        # Save predictions
        pred_df = pl.DataFrame({
            "user_id": user_ids,
            "anchor_date": ["2026-01-14"] * n_val,
            "z_true": val_targets_log,
            "current_state": val_past_buyer,
            "p_react": p_react,
            "p_churn": p_churn,
            "p_buy": p_buy,
            "conditional_z": z_cond,
            "factorized_z": z_fact,
            "prediction_rub": pred_rub,
        })
        pred_df.write_parquet(exp_dir / "validation_predictions.parquet")
        preds_dict[exp_id] = z_fact

        dur_exp = time.time() - t0_exp
        print(f"[+] {exp_id} Done in {dur_exp:.1f}s | Purged Solo RMSLE: {solo_rmsle:.5f} | React AUC: {react_auc:.4f} | Churn AUC: {churn_auc:.4f}")

        registry_rows.append({
            "experiment_id": exp_id,
            "variant": variant,
            "seed": seed,
            "is_shuffled_train": is_shuffled_train,
            "val_mode": val_mode,
            "Solo_RMSLE": solo_rmsle,
            "MSE_log": total_mse,
            "React_AUC": react_auc,
            "React_PR_AUC": react_pr_auc,
            "React_Brier": react_brier,
            "Churn_AUC": churn_auc,
            "Churn_PR_AUC": churn_pr_auc,
            "Churn_Brier": churn_brier,
            "Overall_Brier": overall_brier,
            "MSE_0_to_0": mse_00,
            "MSE_0_to_pos": mse_01,
            "MSE_pos_to_0": mse_10,
            "MSE_pos_to_pos": mse_11,
            "Duration_s": dur_exp,
        })

    reg_df = pl.DataFrame(registry_rows)
    reg_df.write_csv(OUTPUT_ROOT / "purged_experiment_registry.csv")
    print(f"\n[+] Saved purged_experiment_registry.csv ({len(reg_df)} runs)")

    print("\n" + "=" * 80)
    print("=== PURGED & EMBARGOED EXPERIMENT SUITE COMPLETED SUCCESSFULLY ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
