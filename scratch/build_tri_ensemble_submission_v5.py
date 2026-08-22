"""End-to-End Submission v5 Generation: Tri-Ensemble Transition Pipeline (CatBoost Transitions + Hierarchical GRU-365 + Patch Transformer-365)."""

import gc
import time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from torch.utils.data import DataLoader

from src.hurdle import get_feature_columns
from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import HierarchicalGRUModel, PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.boosting import train_churn_classifier, train_reactivation_classifier
from src.transitions.features import compute_all_transition_features
from src.transitions.inference import assemble_factorized_probabilities, compute_factorized_gmv

TEST_ANCHOR = date(2026, 2, 13)
TEST_PARQUET = Path("data/test.parquet")
SUBMISSION_PATH = Path("data/submission.csv")



def main():
    print("===================================================================")
    print("=== BUILDING SUBMISSION V5: TRI-ENSEMBLE TRANSITIONS PIPELINE ===")
    print("===================================================================")
    t_start = time.time()

    # 1. Load Data and User IDs
    data = pl.read_parquet(TRAIN_PARQUET)
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)
    print(f"[*] Total Test Users: {n_test:,}")


    # Training anchors across past year
    from src.snapshots import get_or_create_selected_users
    anchors = generate_panel_anchors()
    train_anchors = anchors[-8:] # 8 training anchors (800k rows)
    train_sample_users = get_or_create_selected_users() # Canonical 100k subset matching snapshots & tensors!


    # -------------------------------------------------------------------------
    # PART 1: CATBOOST TRANSITIONS (A3)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 1] CATBOOST TRANSITIONS TRAINING & INFERENCE ---")
    print("-------------------------------------------------------------------")

    # A. Compute Transition Features for Test
    print("[*] Computing 65 Lifecycle/Last-Year features for Test anchor (2026-02-13)...")
    test_trans_df = compute_all_transition_features(data, test_users, TEST_ANCHOR)
    test_trans_cols = [c for c in test_trans_df.columns if c != "user_id"]

    # Compute Base Snapshot Features for Test (368 cols)
    print("[*] Computing 368 Base Tabular features for Test anchor...")
    test_base_snap = build_snapshot(data, test_users, TEST_ANCHOR, is_test=True)

    all_old_cols = get_feature_columns(test_base_snap)
    noisy_cols = [c for c in all_old_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    old_feat_cols = [c for c in all_old_cols if c not in noisy_cols]

    # Combine Test Matrices
    X_test_old = test_base_snap.select(old_feat_cols).to_numpy().astype(np.float32)
    X_test_trans = test_trans_df.select(test_trans_cols).to_numpy().astype(np.float32)
    X_test_all = np.hstack([X_test_old, X_test_trans])
    past_gmv_test = test_base_snap["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_test = (past_gmv_test > 0).astype(np.int32)
    print(f"[+] Test Feature Matrix Shape: {X_test_all.shape} | Dormant: {np.sum(past_buyer_test == 0):,} | Active: {np.sum(past_buyer_test == 1):,}")

    # B. Assemble Training Matrices across 8 anchors
    print("[*] Computing Training Features and assembling matrix across 8 anchors...")
    X_tr_list, y_tr_list, past_buyer_tr_list = [], [], []
    for a in train_anchors:
        snap_a = pl.read_parquet(f"data/snapshots/snapshot_{a.strftime('%Y-%m-%d')}.parquet")
        trans_a = compute_all_transition_features(data, train_sample_users, a)
        X_old = snap_a.select(old_feat_cols).to_numpy().astype(np.float32)
        X_trans = trans_a.select(test_trans_cols).to_numpy().astype(np.float32)
        X_tr_list.append(np.hstack([X_old, X_trans]))
        y_tr_list.append(snap_a["target"].to_numpy().astype(np.float32))
        past_buyer_tr_list.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
        del snap_a, trans_a

    X_tr_all = np.vstack(X_tr_list)
    y_tr_all = np.concatenate(y_tr_list)
    past_buyer_tr = np.concatenate(past_buyer_tr_list)
    fut_buyer_tr = (y_tr_all > 0).astype(np.int32)
    del X_tr_list, y_tr_list, past_buyer_tr_list
    gc.collect()

    # Masks
    mask_dormant_tr = (past_buyer_tr == 0)
    mask_active_tr = (past_buyer_tr == 1)
    mask_dormant_test = (past_buyer_test == 0)
    mask_active_test = (past_buyer_test == 1)

    # Train CatBoost Reactivation Classifier
    print("[*] Training CatBoost Reactivation Classifier (on dormant users)...")
    clf_react, _ = train_reactivation_classifier(X_tr_all[mask_dormant_tr], fut_buyer_tr[mask_dormant_tr], iterations=700, learning_rate=0.065, verbose=False)
    p_react_test = clf_react.predict_proba(X_test_all[mask_dormant_test])[:, 1]

    # Train CatBoost Churn Classifier
    print("[*] Training CatBoost Churn Classifier (on active users)...")
    clf_churn, _ = train_churn_classifier(X_tr_all[mask_active_tr], (1 - fut_buyer_tr)[mask_active_tr], iterations=700, learning_rate=0.065, verbose=False)
    p_churn_test = clf_churn.predict_proba(X_test_all[mask_active_test])[:, 1]

    # Assemble CatBoost p_buy
    p_buy_cb_test = np.zeros(n_test, dtype=np.float32)
    p_buy_cb_test[mask_dormant_test] = p_react_test
    p_buy_cb_test[mask_active_test] = 1.0 - p_churn_test

    # Train CatBoost Conditional & Direct Regressors
    print("[*] Training CatBoost Conditional Regressor (on buyers)...")
    tr_buyers_mask = (y_tr_all > 0)
    cond_reg = CatBoostRegressor(iterations=700, depth=6, learning_rate=0.065, l2_leaf_reg=5.0, thread_count=4, loss_function="RMSE", random_seed=42, verbose=False)
    cond_reg.fit(X_tr_all[tr_buyers_mask], np.log1p(y_tr_all[tr_buyers_mask]))
    z_cond_cb_test = cond_reg.predict(X_test_all).astype(np.float32)

    print("[*] Training CatBoost Direct Regressor...")
    direct_reg = CatBoostRegressor(iterations=700, depth=6, learning_rate=0.065, l2_leaf_reg=5.0, thread_count=4, loss_function="RMSE", random_seed=42, verbose=False)
    direct_reg.fit(X_tr_all, np.log1p(y_tr_all))
    z_direct_cb_test = direct_reg.predict(X_test_all).astype(np.float32)

    z_catboost_final = (0.50 * z_direct_cb_test + 0.50 * (np.power(p_buy_cb_test, 1.1) * z_cond_cb_test)).astype(np.float32)
    print(f"[+] CatBoost Component Computed | Mean log: {np.mean(z_catboost_final):.3f} | Mean rub: {np.mean(np.expm1(z_catboost_final)):.2f}")

    del X_tr_all, y_tr_all, past_buyer_tr, fut_buyer_tr, X_test_all, X_test_old, X_test_trans
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 2 & 3: LONG-SEQUENCE NEURAL ENCODERS (Zero-RAM Memmap Streaming)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PARTS 2 & 3] HIERARCHICAL GRU & PATCH TRANSFORMER ---")
    print("-------------------------------------------------------------------")

    # Sequence Tensor for 250k Test Users (365d)
    print("[*] Ensuring 365-day daily sequence tensor exists for 250k Test Users...")
    test_tensor_path = CACHE_DIR / f"seq_tensor_2026-02-13_u{n_test}_t365.npy"
    if not test_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, test_users, TEST_ANCHOR, seq_len=365)

    X_test_365_raw = np.load(test_tensor_path, mmap_mode="r")
    scaler_365 = SequentialScaler().fit(X_test_365_raw[:25000])

    # Sequence Training Paths (6 anchors of 100k users)
    seq_train_anchors = anchors[-7:-1]
    train_paths_365, y_tr_seq_list, past_b_tr_seq_list = [], [], []
    for a in seq_train_anchors:
        t_path = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
        if not t_path.exists():
            _ = get_cached_sequence_tensor(data, train_sample_users, a, seq_len=365)
        train_paths_365.append(t_path)
        y_tr_seq_list.append(extract_anchor_targets(data, train_sample_users, a))
        snap_a = pl.read_parquet(f"data/snapshots/snapshot_{a.strftime('%Y-%m-%d')}.parquet")
        past_b_tr_seq_list.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
        del snap_a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader for 365d
    from scratch.run_experiment_long_sequences import MemmapTransitionSequenceDataset, train_transition_seq_model
    train_ds_365 = MemmapTransitionSequenceDataset(train_paths_365, y_tr_seq_list, past_b_tr_seq_list, scaler=scaler_365, seq_len=365)
    train_loader_365 = DataLoader(train_ds_365, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    # Train Hierarchical GRU-365
    print("\n[*] Training Hierarchical GRU-365 on GPU (6 epochs)...")
    hier_model = HierarchicalGRUModel(input_dim=15, hidden_daily=96, hidden_weekly=64).to(device)
    train_transition_seq_model(hier_model, train_loader_365, epochs=6, lr=1e-3, device=device, name="Hierarchical-GRU")

    # Train Patch Transformer-365
    print("\n[*] Training Patch Transformer-365 on GPU (6 epochs)...")
    tf_model = PatchTransformer365Model(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)
    train_transition_seq_model(tf_model, train_loader_365, epochs=6, lr=1e-3, device=device, name="PatchTransformer-365")

    # Inference on 250k Test Users in batched stream
    print("\n[*] Inferring 250,000 test users through Hierarchical GRU & Patch Transformer...")
    hier_model.eval()
    tf_model.eval()

    z_hier_list, z_tf_list = [], []
    inf_batch_size = 1024

    with torch.no_grad():
        for i in range(0, n_test, inf_batch_size):
            raw_batch = X_test_365_raw[i : i + inf_batch_size]
            scaled_batch = (raw_batch - scaler_365.mean) / scaler_365.std
            xb = torch.from_numpy(scaled_batch.astype(np.float32)).float().to(device)

            # Hierarchical GRU
            lr_h, lc_h, _, zc_h, zd_h, _ = hier_model(xb)
            pr_h = torch.sigmoid(lr_h).cpu().numpy()
            pc_h = torch.sigmoid(lc_h).cpu().numpy()
            zc_h = zc_h.cpu().numpy()
            zd_h = zd_h.cpu().numpy()

            pb_batch = past_buyer_test[i : i + inf_batch_size]
            p_buy_h = np.where(pb_batch == 0, pr_h, 1.0 - pc_h)
            z_hier_batch = 0.50 * zd_h + 0.50 * (np.power(p_buy_h, 1.1) * zc_h)
            z_hier_list.append(z_hier_batch)

            # Patch Transformer
            lr_t, lc_t, _, zc_t, zd_t, _ = tf_model(xb)
            pr_t = torch.sigmoid(lr_t).cpu().numpy()
            pc_t = torch.sigmoid(lc_t).cpu().numpy()
            zc_t = zc_t.cpu().numpy()
            zd_t = zd_t.cpu().numpy()

            p_buy_t = np.where(pb_batch == 0, pr_t, 1.0 - pc_t)
            z_tf_batch = 0.50 * zd_t + 0.50 * (np.power(p_buy_t, 1.1) * zc_t)
            z_tf_list.append(z_tf_batch)

    z_hier_final = np.concatenate(z_hier_list).astype(np.float32)
    z_tf_final = np.concatenate(z_tf_list).astype(np.float32)

    # -------------------------------------------------------------------------
    # PART 4: FINAL TRI-ENSEMBLE BLENDING & SUBMISSION
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 4] TRI-ENSEMBLE BLENDING (30% CB + 50% GRU + 20% TF) ---")
    print("-------------------------------------------------------------------")

    z_final = (0.30 * z_catboost_final + 0.50 * z_hier_final + 0.20 * z_tf_final).astype(np.float32)
    y_pred_final = np.clip(np.expm1(z_final), 0.0, None)

    # Sanity checks
    assert len(y_pred_final) == 250000, "Must be exactly 250,000 predictions!"
    assert not np.isnan(y_pred_final).any(), "No NaN allowed!"
    assert not np.isinf(y_pred_final).any(), "No Inf allowed!"

    sub_df = pl.DataFrame({
        "user_id": test_users,
        "predict": y_pred_final,
    })

    sub_df.write_csv(SUBMISSION_PATH)
    elapsed = time.time() - t_start

    print(f"\n[+] SUBMISSION V5 SUCCESSFULLY GENERATED in {elapsed/60:.2f} min!")
    print(f"[+] Output File: {SUBMISSION_PATH.resolve()} ({sub_df.height:,} rows)")
    print("-------------------------------------------------------------------")
    print(f"  - Минимальный прогноз:  {float(np.min(y_pred_final)):.2f} руб.")
    print(f"  - 10-й перцентиль (P10): {float(np.percentile(y_pred_final, 10)):.2f} руб.")
    print(f"  - Медиана (P50):         {float(np.median(y_pred_final)):.2f} руб.")
    print(f"  - Среднее (Mean):        {float(np.mean(y_pred_final)):.2f} руб.")
    print(f"  - 90-й перцентиль (P90): {float(np.percentile(y_pred_final, 90)):.2f} руб.")
    print(f"  - 99-й перцентиль (P99): {float(np.percentile(y_pred_final, 99)):.2f} руб.")
    print(f"  - Максимальный прогноз:  {float(np.max(y_pred_final)):.2f} руб.")
    print("===================================================================")


if __name__ == "__main__":
    main()
