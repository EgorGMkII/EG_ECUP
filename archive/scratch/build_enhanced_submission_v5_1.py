"""Rock-Solid Submission v5.1 Generation: CatBoost Transitions (65%) + Hierarchical GRU-365 Direct (35%) with Strict Quantile Scale Guardrails."""

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
from src.sequential.models import HierarchicalGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.boosting import train_churn_classifier, train_reactivation_classifier
from src.transitions.features import compute_all_transition_features

TEST_ANCHOR = date(2026, 2, 13)
SUBMISSION_PATH = Path("data/submission.csv")


def main():
    print("===================================================================")
    print("=== BUILDING ROCK-SOLID SUBMISSION V5.1 ===")
    print("=== (CatBoost Transitions 65% + Hierarchical GRU-365 35%) ===")
    print("===================================================================")
    t_start = time.time()

    # 1. Load Data and User IDs
    data = pl.read_parquet(TRAIN_PARQUET)
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)
    print(f"[*] Total Test Users: {n_test:,}")

    # Training anchors & canonical 100k users
    anchors = generate_panel_anchors()
    train_anchors = anchors[-8:] # 8 training anchors (800k rows)
    train_sample_users = get_or_create_selected_users()

    # -------------------------------------------------------------------------
    # PART 1: CATBOOST TRANSITIONS (A3) - 65% WEIGHT
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

    cb_cache_path = Path("artifacts/transitions/catboost_test_pred_v5_1.npy")
    cb_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cb_cache_path.exists():
        print(f"[*] Loading cached CatBoost test prediction from {cb_cache_path}...")
        z_catboost_final = np.load(cb_cache_path)
    else:
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
        np.save(cb_cache_path, z_catboost_final)
        del X_tr_all, y_tr_all, past_buyer_tr, fut_buyer_tr

    pred_cb_rub = np.expm1(z_catboost_final)
    print(f"[+] CatBoost Component Ready | Mean rub: {np.mean(pred_cb_rub):.2f} | P50: {np.median(pred_cb_rub):.2f} | P99: {np.percentile(pred_cb_rub, 99):.2f}")
    del X_test_all, X_test_old, X_test_trans
    gc.collect()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # PART 2: PROVEN MULTI-TASK GRU (BCE on All + MSE on Buyers)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 2] PROVEN MULTI-TASK GRU TRAINING (HURDLE LOSS) ---")
    print("-------------------------------------------------------------------")

    from src.sequential.models import MultiTaskGRUModel
    from src.sequential.trainer import train_multitask_gru
    from src.sequential.dataset import MemmapMultiAnchorDataset

    test_tensor_path = CACHE_DIR / f"seq_tensor_2026-02-13_u{len(test_users)}_t90.npy"
    if not test_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, test_users, TEST_ANCHOR, seq_len=90)


    # 14 training anchors for deep coverage
    recent_anchors = anchors[-14:]
    train_paths, train_targets = [], []
    for a in recent_anchors:
        filename = f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t90.npy"
        p = CACHE_DIR / filename
        if not p.exists():
            _ = get_cached_sequence_tensor(data, train_sample_users, a, seq_len=90)
        train_paths.append(p)
        train_targets.append(extract_anchor_targets(data, train_sample_users, a))

    sample_tensor = np.load(train_paths[-1])
    scaler = SequentialScaler().fit(sample_tensor)
    del sample_tensor

    gru_train_ds = MemmapMultiAnchorDataset(train_paths, train_targets, scaler=scaler)
    gru_model = MultiTaskGRUModel(input_dim=15, hidden_dim=128, num_layers=2)
    train_multitask_gru(gru_model, gru_train_ds, epochs=10, batch_size=512, lambda_cls=0.5, lambda_reg=1.0, verbose=True)
    gru_model.eval()

    # Predict on 250k test users
    print("\n[*] Inferring 250,000 test users through MultiTask GRU...")
    test_tensor_raw = np.load(test_tensor_path)
    test_tensor_scaled = scaler.transform(test_tensor_raw)
    del test_tensor_raw

    p_gru_list, z_cond_gru_list = [], []
    inf_bs = 1024
    with torch.no_grad():
        for i in range(0, len(test_tensor_scaled), inf_bs):
            batch_x = torch.from_numpy(test_tensor_scaled[i : i + inf_bs]).float().to(device)
            p_logits_t, z_cond_t, _ = gru_model(batch_x)
            p_gru_list.append(torch.sigmoid(p_logits_t).cpu().numpy())
            z_cond_gru_list.append(torch.clamp(z_cond_t, min=0.0).cpu().numpy())

    p_gru = np.concatenate(p_gru_list)
    z_cond_gru = np.concatenate(z_cond_gru_list)
    z_gru_final = (np.power(p_gru, 1.1) * z_cond_gru).astype(np.float32)

    pred_gru_rub = np.expm1(z_gru_final)
    print(f"[+] MultiTask GRU Computed | Mean rub: {np.mean(pred_gru_rub):.2f} | P50: {np.median(pred_gru_rub):.2f} | P99: {np.percentile(pred_gru_rub, 99):.2f}")
    del test_tensor_scaled, gru_model, gru_train_ds
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 3: ENSEMBLE BLENDING (50% CatBoost Transitions + 50% MultiTask GRU)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 3] BLENDING (50% CatBoost Transitions + 50% MultiTask GRU) ---")
    print("-------------------------------------------------------------------")

    z_final = (0.50 * z_catboost_final + 0.50 * z_gru_final).astype(np.float32)
    y_pred_final = np.clip(np.expm1(z_final), 0.0, None)

    # Scale Guardrails
    mean_val = float(np.mean(y_pred_final))
    p50_val = float(np.median(y_pred_final))
    p90_val = float(np.percentile(y_pred_final, 90))
    p99_val = float(np.percentile(y_pred_final, 99))
    max_val = float(np.max(y_pred_final))

    print(f"\n[GUARDRAIL AUDIT]:")
    print(f"  - Count: {len(y_pred_final):,}")
    print(f"  - Min:   {float(np.min(y_pred_final)):.2f} руб.")
    print(f"  - P50:   {p50_val:.2f} руб. (Target: ~6-8 руб.)")
    print(f"  - Mean:  {mean_val:.2f} руб. (Target: ~32-42 руб.)")
    print(f"  - P90:   {p90_val:.2f} руб. (Target: ~90-120 руб.)")
    print(f"  - P99:   {p99_val:.2f} руб. (Target: ~400-520 руб.)")
    print(f"  - Max:   {max_val:.2f} руб. (Target: ~2500-4500 руб.)")

    assert len(y_pred_final) == 250000, "Must be exactly 250,000 predictions!"
    assert not np.isnan(y_pred_final).any(), "No NaN allowed!"
    assert not np.isinf(y_pred_final).any(), "No Inf allowed!"
    assert mean_val >= 25.0, f"Mean too low: {mean_val:.2f} rub!"
    assert p99_val >= 300.0, f"P99 too low: {p99_val:.2f} rub!"

    sub_df = pl.DataFrame({
        "user_id": test_users,
        "predict": y_pred_final,
    })

    sub_df.write_csv(SUBMISSION_PATH)
    elapsed = time.time() - t_start

    print(f"\n[+] SUBMISSION V5.1 SUCCESSFULLY SAVED to {SUBMISSION_PATH.resolve()} in {elapsed/60:.2f} min!")
    print("===================================================================")



if __name__ == "__main__":
    main()
