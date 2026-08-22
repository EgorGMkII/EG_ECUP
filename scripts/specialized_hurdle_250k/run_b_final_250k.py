"""RUN B — FINAL_250K: Final Training on ALL 23 Anchors & Final Submission Inference.

1. Cutoff: target_end(anchor) <= 2026-02-13 (All 23 anchors: 2025-03-31 .. 2026-01-14).
2. Inference Anchor: 2026-02-13 (target: 2026-02-14 .. 2026-03-15).
3. Models trained from scratch:
   - CatBoost: CB_REACT_FINAL_250K, CB_CHURN_FINAL_250K, CB_AMOUNT_FINAL_250K.
   - S1 Masked GRU Specialists.
   - S2 Dense GRU Specialists.
   - ETT Specialists (180 tok, tau=30d).
4. Raw Test Prediction Bank: test_specialists_raw_predictions_250k_v2.parquet.
5. Applies Joint Meta Package from RUN A (joint_meta_weights_250k.json).
6. Exports final submission: submission_specialized_hurdle_joint_250k_v2.csv.
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.snapshots import build_snapshot
from scripts.specialized_hurdle_250k.run_a_meta_250k import (
    GRUEncoder,
    EventTimeTransformer,
    MemmapDataset,
    extract_event_time_sequences,
)


def main():
    print("=" * 80)
    print("RUN B — FINAL_250K: FINAL RETRAINING & SUBMISSION INFERENCE (ALL 250k USERS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path("artifacts/specialized_hurdle_250k/final_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = Path("artifacts/specialized_hurdle_250k/submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load User Cohort (250,000 users)
    df_sample = pl.read_csv("sample_submit.csv")
    all_users = df_sample["user_id"].to_numpy()
    n_users = len(all_users)
    print(f"[+] Loaded 250k user cohort: {n_users:,} unique users.")

    # 2. RUN B Temporal Anchors (All 23 Anchors)
    test_anchor = "2026-02-13"
    run_b_cb_anchors = [
        "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
        "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
        "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
        "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15",
        "2025-12-22", "2026-01-05", "2026-01-14"
    ]
    run_b_neural_anchors = [
        "2025-03-31", "2025-04-28", "2025-05-26", "2025-06-23",
        "2025-07-21", "2025-08-18", "2025-09-15", "2026-01-14"
    ]
    print(f"[+] RUN B CatBoost training anchors ({len(run_b_cb_anchors)}): {run_b_cb_anchors[0]} .. {run_b_cb_anchors[-1]}")
    print(f"[+] RUN B Neural training anchors ({len(run_b_neural_anchors)}): {run_b_neural_anchors}")
    print(f"[+] Final Test Anchor: {test_anchor}")

    # 3. Load RUN A Joint Meta Package
    meta_paths = [
        Path("artifacts/specialized_hurdle_250k/meta_run/joint_meta_weights_250k.json"),
        Path("joint_meta_weights_250k.json"),
        Path("artifacts/specialized_hurdle_250k/joint_meta/joint_meta_weights_250k.json"),
    ]
    meta_pkg = None
    for p in meta_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                meta_pkg = json.load(f)
            print(f"[+] Loaded RUN A Joint Meta Package from {p}")
            break
            
    if meta_pkg is None:
        raise FileNotFoundError("Could not find joint_meta_weights_250k.json from RUN A!")

    w_react = np.array(meta_pkg["react_stack_weights"], dtype=np.float64)
    w_churn = np.array(meta_pkg["churn_stack_weights"], dtype=np.float64)
    w_amount = np.array(meta_pkg["amount_ridge_coefficients"], dtype=np.float64)
    b_amount = float(meta_pkg["amount_ridge_intercept"])
    scaler_cfg = meta_pkg["amount_scaler"]
    scaler_means = np.array(scaler_cfg["means"], dtype=np.float64)
    scaler_scales = np.array(scaler_cfg["scales"], dtype=np.float64)

    print(f"[*] React Stack Weights:     {w_react}")
    print(f"[*] Churn Stack Weights:     {w_churn}")
    print(f"[*] Amount Ridge Coeffs:     {w_amount} + {b_amount}")

    # 4. Load Raw Events
    events_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    print(f"\n[*] Loading raw events from {events_path}...")
    df_events = pl.read_parquet(events_path)
    print(f"[+] Loaded raw events: {len(df_events):,} rows.")

    snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")

    # 5. STEP A: FINAL CATBOOST SPECIALISTS (2.3M ROWS)
    print("\n" + "=" * 80)
    print("STEP A: TRAINING FINAL CATBOOST SPECIALISTS (ALL 23 ANCHORS)")
    print("=" * 80)

    cb_dfs = []
    for anc in run_b_cb_anchors:
        anc_path = snap_dir / f"snapshot_{anc}.parquet"
        if not anc_path.exists():
            anc_path = Path(f"anchor_{anc}.parquet")
        df_snap = pl.read_parquet(anc_path)
        cb_dfs.append(df_snap)

    df_cb_train = pl.concat(cb_dfs)
    print(f"[+] Pooled CatBoost training data for RUN B: {len(df_cb_train):,} rows.")

    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d", "anchor_date", "history_start", "history_end", "target_start", "target_end", "user_segment_id"}
    feat_cols = [c for c in df_cb_train.columns if c not in excluded]
    
    y_train_cb_gmv = df_cb_train["target"].to_numpy().astype(np.float32)
    was_act_tr = (df_cb_train["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_tr = (y_train_cb_gmv > 0).astype(int)
    X_cb_tr = df_cb_train.select(feat_cols).to_numpy().astype(np.float32)

    # CB_REACT
    mask_react = (was_act_tr == 0)
    print(f"\n[*] Training CB_REACT_FINAL_250K on {mask_react.sum():,} rows...")
    cb_react = CatBoostClassifier(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, task_type="GPU", random_seed=42, verbose=500)
    cb_react.fit(X_cb_tr[mask_react], will_buy_tr[mask_react])

    # CB_CHURN
    mask_churn = (was_act_tr == 1)
    print(f"\n[*] Training CB_CHURN_FINAL_250K on {mask_churn.sum():,} rows...")
    cb_churn = CatBoostClassifier(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, task_type="GPU", random_seed=42, verbose=500)
    cb_churn.fit(X_cb_tr[mask_churn], 1 - will_buy_tr[mask_churn])

    # CB_AMOUNT
    mask_amt = (y_train_cb_gmv > 0)
    print(f"\n[*] Training CB_AMOUNT_FINAL_250K on {mask_amt.sum():,} rows...")
    cb_amount = CatBoostRegressor(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, loss_function="RMSE", eval_metric="RMSE", task_type="GPU", random_seed=42, verbose=500)
    cb_amount.fit(X_cb_tr[mask_amt], np.log1p(y_train_cb_gmv[mask_amt]))

    del df_cb_train, X_cb_tr, was_act_tr, will_buy_tr, y_train_cb_gmv, cb_dfs
    gc.collect()

    # 6. STEP B: FINAL NEURAL SPECIALISTS (S1, S2, ETT)
    print("\n" + "=" * 80)
    print("STEP B: TRAINING FINAL NEURAL SPECIALISTS (RUN B)")
    print("=" * 80)

    n_neural_anchors = len(run_b_neural_anchors)
    total_seq_samples = n_neural_anchors * 100000

    train_c = np.zeros((total_seq_samples, 180, 12), dtype=np.float32)
    train_t = np.zeros((total_seq_samples, 180, 12), dtype=np.float32)
    train_r = np.zeros((total_seq_samples, 180), dtype=np.int64)
    train_m = np.ones((total_seq_samples, 180), dtype=bool)
    train_emp = np.ones(total_seq_samples, dtype=bool)
    train_z = np.zeros(total_seq_samples, dtype=np.float32)
    train_act = np.zeros(total_seq_samples, dtype=np.float32)
    train_wb = np.zeros(total_seq_samples, dtype=np.float32)
    train_rub = np.zeros(total_seq_samples, dtype=np.float32)

    for i, anc in enumerate(run_b_neural_anchors):
        print(f"[*] Extracting sequences for {anc} ({i+1}/{n_neural_anchors})...")
        anc_path = snap_dir / f"snapshot_{anc}.parquet"
        if not anc_path.exists():
            anc_path = Path(f"anchor_{anc}.parquet")
        df_snap_anc = pl.read_parquet(anc_path)
        snap_users = df_snap_anc["user_id"].to_numpy()
        
        offset = i * 100000
        extract_event_time_sequences(
            df_events, snap_users, anc, max_events=180, tau_days=30.0,
            out_c=train_c, out_t=train_t, out_r=train_r, out_m=train_m, out_emp=train_emp,
            offset=offset
        )
        
        rub = df_snap_anc["target"].to_numpy().astype(np.float32)
        train_rub[offset:offset+100000] = rub
        train_act[offset:offset+100000] = (df_snap_anc["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(np.float32)
        train_wb[offset:offset+100000] = (rub > 0).astype(np.float32)
        train_z[offset:offset+100000] = np.log1p(np.maximum(rub, 0.0))

    dataset = MemmapDataset(train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_wb, train_rub)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=2)
    print(f"[+] Total neural training sequences: {len(dataset):,} samples.")

    # S1
    print("\n[*] Training Final S1 Masked GRU Specialists...")
    s1_model = GRUEncoder(d_in=12, d_model=128, n_layers=2, dropout=0.10).to(device)
    opt_s1 = torch.optim.AdamW(s1_model.parameters(), lr=3e-4, weight_decay=1e-4)
    s1_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(s1_iter)
        except StopIteration: s1_iter = iter(loader); batch = next(s1_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        c_masked = torch.where(m.unsqueeze(-1), torch.zeros_like(c), c)
        dir_z, cond_z, r_log, c_log = s1_model(c_masked, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_s1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1_model.parameters(), 1.0); opt_s1.step()

    # S2
    print("\n[*] Training Final S2 Dense GRU Specialists...")
    s2_model = GRUEncoder(d_in=12, d_model=128, n_layers=2, dropout=0.10).to(device)
    opt_s2 = torch.optim.AdamW(s2_model.parameters(), lr=3e-4, weight_decay=1e-4)
    s2_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(s2_iter)
        except StopIteration: s2_iter = iter(loader); batch = next(s2_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        dir_z, cond_z, r_log, c_log = s2_model(c, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_s2.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s2_model.parameters(), 1.0); opt_s2.step()

    # ETT
    print("\n[*] Training Final Event-Time Transformer Specialists (180 tok, tau=30d)...")
    ett_model = EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=180).to(device)
    opt_ett = torch.optim.AdamW(ett_model.parameters(), lr=3e-4, weight_decay=1e-4)
    ett_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(ett_iter)
        except StopIteration: ett_iter = iter(loader); batch = next(ett_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        dir_z, cond_z, r_log, c_log = ett_model(c, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_ett.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ett_model.parameters(), 1.0); opt_ett.step()

    del train_c, train_t, train_r, train_m, train_emp, train_act, train_wb, train_rub, train_z, loader, dataset
    gc.collect()

    # 7. STEP C: TEST INFERENCE (2026-02-13) ON ALL 250,000 USERS
    print("\n" + "=" * 80)
    print("STEP C: TEST INFERENCE ON ANCHOR 2026-02-13 (ALL 250,000 USERS)")
    print("=" * 80)

    # CatBoost Inference on Test Anchor
    print(f"[*] Building 250k Test Feature Snapshot for {test_anchor}...")
    df_test_snap = build_snapshot(
        data=df_events,
        user_ids=all_users.tolist(),
        anchor_date=pd.Timestamp(test_anchor).date(),
        is_test=True,
    )
    X_test = df_test_snap.select(feat_cols).to_numpy().astype(np.float32)
    was_act_test = (df_test_snap["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int) if "lifetime_gmv" in df_test_snap.columns else np.ones(n_users, dtype=int)

    cb_r_logit = cb_react.predict(X_test, prediction_type="RawFormulaVal")
    cb_c_logit = cb_churn.predict(X_test, prediction_type="RawFormulaVal")
    cb_a_z = cb_amount.predict(X_test)

    # Neural Inference on Test Anchor
    print(f"[*] Extracting Test Sequences for {test_anchor} (250,000 users)...")
    t_c = np.zeros((n_users, 180, 12), dtype=np.float32)
    t_t = np.zeros((n_users, 180, 12), dtype=np.float32)
    t_r = np.zeros((n_users, 180), dtype=np.int64)
    t_m = np.ones((n_users, 180), dtype=bool)
    t_emp = np.ones(n_users, dtype=bool)

    extract_event_time_sequences(
        df_events, all_users, test_anchor, max_events=180, tau_days=30.0,
        out_c=t_c, out_t=t_t, out_r=t_r, out_m=t_m, out_emp=t_emp, offset=0
    )

    s1_r_list, s1_c_list, s1_a_list = [], [], []
    s2_r_list, s2_c_list, s2_a_list = [], [], []
    ett_r_list, ett_c_list, ett_a_list = [], [], []

    bs = 1024
    s1_model.eval(); s2_model.eval(); ett_model.eval()

    with torch.no_grad():
        for i in range(0, n_users, bs):
            end_i = min(i + bs, n_users)
            c_b = torch.from_numpy(t_c[i:end_i]).to(device)
            t_b = torch.from_numpy(t_t[i:end_i]).to(device)
            r_b = torch.from_numpy(t_r[i:end_i]).to(device)
            m_b = torch.from_numpy(t_m[i:end_i]).to(device)
            emp_b = torch.from_numpy(t_emp[i:end_i]).to(device)

            c_b_masked = torch.where(m_b.unsqueeze(-1), torch.zeros_like(c_b), c_b)
            _, s1_a, s1_r, s1_c = s1_model(c_b_masked, t_b, r_b, m_b, emp_b)
            s1_r_list.append(s1_r.cpu().numpy()); s1_c_list.append(s1_c.cpu().numpy()); s1_a_list.append(s1_a.cpu().numpy())

            _, s2_a, s2_r, s2_c = s2_model(c_b, t_b, r_b, m_b, emp_b)
            s2_r_list.append(s2_r.cpu().numpy()); s2_c_list.append(s2_c.cpu().numpy()); s2_a_list.append(s2_a.cpu().numpy())

            _, ett_a, ett_r, ett_c = ett_model(c_b, t_b, r_b, m_b, emp_b)
            ett_r_list.append(ett_r.cpu().numpy()); ett_c_list.append(ett_c.cpu().numpy()); ett_a_list.append(ett_a.cpu().numpy())

    s1_r_logit = np.concatenate(s1_r_list); s1_c_logit = np.concatenate(s1_c_list); s1_a_z = np.concatenate(s1_a_list)
    s2_r_logit = np.concatenate(s2_r_list); s2_c_logit = np.concatenate(s2_c_list); s2_a_z = np.concatenate(s2_a_list)
    ett_r_logit = np.concatenate(ett_r_list); ett_c_logit = np.concatenate(ett_c_list); ett_a_z = np.concatenate(ett_a_list)

    # Save Test Raw Prediction Bank (v2)
    df_raw_v2 = pl.DataFrame({
        "user_id": all_users,
        "was_active": was_act_test,
        "cb_react_logit": cb_r_logit,
        "s1_react_logit": s1_r_logit,
        "s2_react_logit": s2_r_logit,
        "ett_react_logit": ett_r_logit,
        "cb_churn_logit": cb_c_logit,
        "s1_churn_logit": s1_c_logit,
        "s2_churn_logit": s2_c_logit,
        "ett_churn_logit": ett_c_logit,
        "cb_amount_z": cb_a_z,
        "s1_amount_z": s1_a_z,
        "s2_amount_z": s2_a_z,
        "ett_amount_z": ett_a_z,
    })
    raw_v2_path = out_dir / "test_specialists_raw_predictions_250k_v2.parquet"
    df_raw_v2.write_parquet(raw_v2_path)
    df_raw_v2.write_parquet(Path("test_specialists_raw_predictions_250k_v2.parquet"))
    print(f"[+] Saved Raw Prediction Bank v2 to {raw_v2_path} and root")

    # 8. STEP D: ASSEMBLE FINAL SUBMISSION WITH JOINT META PACKAGE
    print("\n" + "=" * 80)
    print("STEP D: COMPUTING FINAL JOINT 250k SUBMISSION")
    print("=" * 80)

    X_r_test = np.column_stack([cb_r_logit, s1_r_logit, s2_r_logit, ett_r_logit])
    X_c_test = np.column_stack([cb_c_logit, s1_c_logit, s2_c_logit, ett_c_logit])
    X_a_test = np.column_stack([cb_a_z, s1_a_z, s2_a_z, ett_a_z])

    p_r_test = expit(X_r_test @ w_react)
    p_c_test = expit(X_c_test @ w_churn)
    p_buy_test = np.where(was_act_test == 0, p_r_test, 1.0 - p_c_test)

    X_a_test_scaled = (X_a_test - scaler_means) / scaler_scales
    cond_z_test = np.clip(X_a_test_scaled @ w_amount + b_amount, 0.0, None)

    z_pred_test = np.power(np.clip(p_buy_test, 0.0, 1.0), 1.1) * cond_z_test
    z_pred_test = np.clip(z_pred_test, 0.0, None)
    gmv_final = np.expm1(z_pred_test)

    # Submission File
    df_sub_v2 = pl.DataFrame({
        "user_id": all_users,
        "predict": gmv_final
    })
    sub_path = Path("submission_specialized_hurdle_joint_250k_v2.csv")
    df_sub_v2.write_csv(sub_path)
    df_sub_v2.write_csv(sub_dir / "submission_specialized_hurdle_joint_250k_v2.csv")
    print(f"[+] FINAL SUBMISSION SAVED TO {sub_path} ({len(df_sub_v2):,} rows)")
    print(f"    Mean GMV:   {np.mean(gmv_final):.2f} RUB")
    print(f"    Median GMV: {np.median(gmv_final):.2f} RUB")
    print(f"    Min GMV:    {np.min(gmv_final):.4f} RUB")
    print(f"    Max GMV:    {np.max(gmv_final):.2f} RUB")


if __name__ == "__main__":
    main()
