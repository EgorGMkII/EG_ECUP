"""Build Final Tri-Ensemble Submission v6.0: CatBoost Transitions (35%) + MultiTask GRU (45%) + T5 PreLN Patch Transformer (20%) with alpha=1.10."""

import gc
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from scratch.run_experiment_long_sequences import MemmapTransitionSequenceDataset, train_transition_seq_model
from scratch.audit_transformer_part4_rehab_suite import SimplifiedTransformerModel

TEST_ANCHOR = date(2026, 2, 13)
SUBMISSION_PATH = Path("data/submission.csv")
TRANSITIONS_ARTIFACTS = Path("artifacts/transitions")


def main():
    print("===================================================================")
    print("=== BUILDING FINAL TRI-ENSEMBLE SUBMISSION V6.0 ===")
    print("=== (CatBoost 35% + MultiTask GRU 45% + T5 PreLN Transformer 20%) ===")
    print("===================================================================")
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pl.read_parquet(TRAIN_PARQUET)
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)
    print(f"[*] Total Test Users: {n_test:,}")

    anchors = generate_panel_anchors()
    train_sample_users = get_or_create_selected_users()

    # 1. Past Buyer status on Test
    test_base_snap = build_snapshot(data, test_users, TEST_ANCHOR, is_test=True)
    past_gmv_test = test_base_snap["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_test = (past_gmv_test > 0).astype(np.int32)
    del test_base_snap
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 1: LOAD CACHED CATBOOST TRANSITIONS A3 (35% WEIGHT)
    # -------------------------------------------------------------------------
    cb_cache_path = TRANSITIONS_ARTIFACTS / "catboost_test_pred_v5_1.npy"
    if not cb_cache_path.exists():
        raise FileNotFoundError(f"{cb_cache_path} not found!")
    z_catboost = np.load(cb_cache_path)
    pred_cb_rub = np.expm1(z_catboost)
    print(f"[+] [1/3] CatBoost Transitions Ready | Mean rub: {np.mean(pred_cb_rub):.2f} | P50: {np.median(pred_cb_rub):.2f} | P99: {np.percentile(pred_cb_rub, 99):.2f}")

    # -------------------------------------------------------------------------
    # PART 2: LOAD CACHED MULTITASK GRU-90 (45% WEIGHT)
    # -------------------------------------------------------------------------
    gru_cache_path = TRANSITIONS_ARTIFACTS / "gru_test_pred_v5_1.npy"
    if not gru_cache_path.exists():
        raise FileNotFoundError(f"{gru_cache_path} not found!")
    z_gru = np.load(gru_cache_path)
    pred_gru_rub = np.expm1(z_gru)
    print(f"[+] [2/3] MultiTask GRU Ready | Mean rub: {np.mean(pred_gru_rub):.2f} | P50: {np.median(pred_gru_rub):.2f} | P99: {np.percentile(pred_gru_rub, 99):.2f}")

    # -------------------------------------------------------------------------
    # PART 3: TRAIN & INFER T5 PRELN PATCH TRANSFORMER-365 (20% WEIGHT)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 3] T5 PRE-LN PATCH TRANSFORMER-365 (SOLO VAL: 1.668) ---")
    print("-------------------------------------------------------------------")

    test_tensor_path_365 = CACHE_DIR / f"seq_tensor_2026-02-13_u{len(test_users)}_t365.npy"
    if not test_tensor_path_365.exists():
        _ = get_cached_sequence_tensor(data, test_users, TEST_ANCHOR, seq_len=365)

    X_test_365_raw = np.load(test_tensor_path_365, mmap_mode="r")
    scaler_365 = SequentialScaler().fit(X_test_365_raw[:25000])

    seq_train_anchors = anchors[-8:]
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

    train_ds_365 = MemmapTransitionSequenceDataset(train_paths_365, y_tr_seq_list, past_b_tr_seq_list, scaler=scaler_365, seq_len=365)
    train_loader_365 = DataLoader(train_ds_365, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    # Simplified 2-Layer PreLN Transformer with high-contrast representation
    t5_model = SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    train_transition_seq_model(t5_model, train_loader_365, epochs=6, lr=1e-3, device=device, name="T5-PreLN-Transformer")
    t5_model.eval()

    # Predict on 250k test users
    print("\n[*] Inferring 250,000 test users through T5 PreLN Transformer...")
    p_react_list, p_churn_list, z_cond_list = [], [], []
    inf_bs = 1024

    with torch.no_grad():
        for i in range(0, n_test, inf_bs):
            raw_batch = X_test_365_raw[i : i + inf_bs]
            scaled_batch = (raw_batch - scaler_365.mean) / scaler_365.std
            xb = torch.from_numpy(scaled_batch.astype(np.float32)).float().to(device)

            lr, lc, _, zc, _, _ = t5_model(xb)
            p_react_list.append(torch.sigmoid(lr).cpu().numpy())
            p_churn_list.append(torch.sigmoid(lc).cpu().numpy())
            z_cond_list.append(zc.cpu().numpy())

    p_react_tf = np.concatenate(p_react_list)
    p_churn_tf = np.concatenate(p_churn_list)
    z_cond_tf = np.concatenate(z_cond_list)

    # Factorized Hurdle probability with verified optimal alpha = 1.10
    p_buy_tf = np.where(past_buyer_test == 0, p_react_tf, 1.0 - p_churn_tf)
    z_tf = (np.power(p_buy_tf, 1.10) * z_cond_tf).astype(np.float32)

    pred_tf_rub = np.expm1(z_tf)
    print(f"[+] [3/3] T5 PreLN Transformer Ready | Mean rub: {np.mean(pred_tf_rub):.2f} | P50: {np.median(pred_tf_rub):.2f} | P99: {np.percentile(pred_tf_rub, 99):.2f}")
    del t5_model, train_ds_365, X_test_365_raw
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 4: TRI-ENSEMBLE BLEND (35% CB + 45% GRU + 20% T5 TF)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 4] TRI-ENSEMBLE BLENDING (35% CB + 45% GRU + 20% TF) ---")
    print("-------------------------------------------------------------------")

    z_final = (0.35 * z_catboost + 0.45 * z_gru + 0.20 * z_tf).astype(np.float32)
    y_pred_final = np.clip(np.expm1(z_final), 0.0, None)

    mean_val = float(np.mean(y_pred_final))
    p50_val = float(np.median(y_pred_final))
    p90_val = float(np.percentile(y_pred_final, 90))
    p99_val = float(np.percentile(y_pred_final, 99))
    max_val = float(np.max(y_pred_final))

    print(f"\n[GUARDRAIL AUDIT FOR SUBMISSION V6.0]:")
    print(f"  - Count: {len(y_pred_final):,}")
    print(f"  - Min:   {float(np.min(y_pred_final)):.2f} руб.")
    print(f"  - P50:   {p50_val:.2f} руб. (Target: ~6-8 руб.)")
    print(f"  - Mean:  {mean_val:.2f} руб. (Target: ~34-42 руб.)")
    print(f"  - P90:   {p90_val:.2f} руб. (Target: ~90-120 руб.)")
    print(f"  - P99:   {p99_val:.2f} руб. (Target: ~400-520 руб.)")
    print(f"  - Max:   {max_val:.2f} руб. (Target: ~2500-4500 руб.)")

    assert len(y_pred_final) == 250000, "Must be exactly 250,000 predictions!"
    assert not np.isnan(y_pred_final).any(), "No NaN allowed!"
    assert not np.isinf(y_pred_final).any(), "No Inf allowed!"
    assert mean_val >= 30.0, f"Mean too low: {mean_val:.2f} rub!"
    assert p99_val >= 380.0, f"P99 too low: {p99_val:.2f} rub!"

    # Save to submission.csv and submission_v6_tri_ensemble.csv
    sub_df = pl.DataFrame({
        "user_id": test_users,
        "predict": y_pred_final,
    })
    sub_df.write_csv(SUBMISSION_PATH)
    sub_df.write_csv(Path("data/submission_v6_tri_ensemble.csv"))
    elapsed = time.time() - t_start

    print(f"\n[+] SUCCESS! TRI-ENSEMBLE V6.0 SAVED to {SUBMISSION_PATH.resolve()} in {elapsed/60:.2f} min!")
    print("===================================================================")


if __name__ == "__main__":
    main()
