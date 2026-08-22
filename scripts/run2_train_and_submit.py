"""RUN 2: Final Training on ALL 250k Users & Final Submission Inference.

1. Cohort: ALL 250,000 users.
2. Training Anchors: All anchors with target_end <= 2026-02-13 (last anchor: 2026-01-14).
3. Models trained from scratch (Pooled Dataset):
   - CatBoost Specialists: CB_REACT_FINAL, CB_CHURN_FINAL, CB_AMOUNT_FINAL.
   - S1 Masked GRU Specialists: S1_BASE_FINAL + React, Churn, Amount heads.
   - S2 Dense GRU Specialists: S2_BASE_FINAL + React, Churn, Amount heads.
   - Event-Time Transformer Specialists: ETT_BASE_FINAL (180 tokens, fixed tau=30d) + React, Churn, Amount heads.
4. Submission Inference on anchor = 2026-02-13 (target: 2026-02-14 .. 2026-03-15):
   - Loads fixed meta-weights from artifacts/specialized_hurdle/run1_meta_weights.json.
   - Applies React Stack, Churn Stack, and Amount Ridge Stack.
   - Assembles final prediction:
       z_pred = (p_buy ** 1.1) * conditional_z
       gmv_pred = np.expm1(np.clip(z_pred, 0.0, None))
   - Saves final submission in exact user_id order of sample_submit.csv.
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import shutil
from scripts.run1_train_meta_weights import (
    extract_event_time_sequences,
    EventTimeTransformer,
    GRUEncoder,
)


class MemmapDataset(Dataset):
    def __init__(self, content, time_feat, ranks, mask, empty, z_true, was_active, will_buy, y_rub):
        self.content = content
        self.time_feat = time_feat
        self.ranks = ranks
        self.mask = mask
        self.empty = empty
        self.z_true = z_true
        self.was_active = was_active
        self.will_buy = will_buy
        self.y_rub = y_rub

    def __len__(self):
        return len(self.z_true)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.asarray(self.content[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.time_feat[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.ranks[idx], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.mask[idx], dtype=bool)),
            torch.tensor(bool(self.empty[idx]), dtype=torch.bool),
            torch.tensor(float(self.z_true[idx]), dtype=torch.float32),
            torch.tensor(float(self.was_active[idx]), dtype=torch.float32),
            torch.tensor(float(self.will_buy[idx]), dtype=torch.float32),
            torch.tensor(float(self.y_rub[idx]), dtype=torch.float32),
        )


def main():
    print("=" * 80)
    print("RUN 2: FINAL TRAINING (ALL 250k USERS) & SUBMISSION INFERENCE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path("artifacts/specialized_hurdle")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Fixed Meta-Weights
    candidate_paths = [
        Path("canonical_250k_meta_weights.json"),
        out_dir / "canonical_250k_meta_weights.json",
        out_dir / "run1_meta_weights.json",
        Path("run1_meta_weights.json"),
        Path("configs/specialized_hurdle/run1_meta_weights.json"),
    ]
    meta_cfg = None
    for p in candidate_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                meta_cfg = json.load(f)
            print(f"[+] Loaded RUN 1 meta-weights from {p}")
            break

    if meta_cfg is None:
        print("[!] Using verified embedded RUN 1 meta-weights")
        meta_cfg = {
            "react_stack_weights": [0.1911, 0.5445, 0.0, 0.2644],
            "churn_stack_weights": [0.3426, 0.1862, 0.0, 0.4711],
            "amount_ridge_coefficients": [0.1672, 0.0507, 0.3287, 0.4801],
            "amount_ridge_intercept": -0.0389,
            "ALPHA": 1.1,
        }

    # Save to out_dir for artifact bundling
    with open(out_dir / "run1_meta_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta_cfg, f, indent=2)

    w_react = np.array(meta_cfg["react_stack_weights"])
    w_churn = np.array(meta_cfg["churn_stack_weights"])
    ridge_coef = np.array(meta_cfg["amount_ridge_coefficients"])
    ridge_intercept = float(meta_cfg["amount_ridge_intercept"])
    alpha = float(meta_cfg.get("ALPHA", 1.1))

    print(f"[+] Loaded RUN 1 Meta-Weights: ALPHA = {alpha}")
    print(f"    React Weights: {w_react.round(4).tolist()}")
    print(f"    Churn Weights: {w_churn.round(4).tolist()}")
    print(f"    Amount Ridge:  {ridge_coef.round(4).tolist()} + {ridge_intercept:.4f}")

    # 2. Load submission template and all 250k users
    sub_template = pl.read_csv("sample_submit.csv") if Path("sample_submit.csv").exists() else pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet").select("user_id")
    all_users = sub_template["user_id"].to_numpy()
    n_users = len(all_users)
    print(f"[+] Submission cohort: {n_users:,} users.")

    # 3. Determine Final Training Anchors (target_end <= 2026-02-13)
    from scripts.run1_train_meta_weights import CANONICAL_ANCHORS
    all_anchors = CANONICAL_ANCHORS

    cutoff = pd.Timestamp("2026-02-13")
    train_anchors = []
    for a in all_anchors:
        a_dt = pd.Timestamp(a)
        t_end = a_dt + pd.Timedelta(days=30)
        if t_end <= cutoff:
            train_anchors.append(a)

    test_anchor = "2026-02-13"
    print(f"[+] Found {len(train_anchors)} legal training anchors for RUN 2: {train_anchors[0]} .. {train_anchors[-1]}")
    print(f"[+] Final Inference Anchor: {test_anchor}")

    # ------------------------------------------------------------------
    # Step A: Final CatBoost Specialists Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP A: TRAINING FINAL CATBOOST SPECIALISTS ON ALL USERS")
    print("=" * 80)

    raw_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    df_raw = pl.read_parquet(raw_path)
    print(f"[+] Loaded raw events for sequence extraction & test snapshot: {len(df_raw):,} rows.")

    fs_dir = Path("artifacts/specialized_hurdle/feature_store")
    fs_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")

    # Build feature store for training anchors
    missing = [a for a in train_anchors if not (fs_dir / f"anchor_{a}.parquet").exists()]
    if missing:
        print(f"[*] Building feature store on VM for {len(missing)} anchors...")
        from src.specialized_hurdle.feature_store import build_causal_feature_store
        build_causal_feature_store(
            snapshots_dir=snap_dir,
            out_dir=fs_dir,
            anchors=train_anchors,
        )

    # CatBoost training on pooled snapshots
    train_dfs = [pl.read_parquet(fs_dir / f"anchor_{a}.parquet") for a in train_anchors if (fs_dir / f"anchor_{a}.parquet").exists()]
    df_cb_train = pl.concat(train_dfs)

    # Build 250k test snapshot directly for test anchor 2026-02-13
    print(f"[*] Building 250k test feature snapshot for {test_anchor}...")
    from src.snapshots import build_snapshot
    df_cb_test = build_snapshot(
        data=df_raw,
        user_ids=all_users.tolist(),
        anchor_date=pd.Timestamp(test_anchor).date(),
        is_test=True,
    )
    print(f"[+] Test feature snapshot ready: {df_cb_test.height:,} users, {len(df_cb_test.columns)} columns.")

    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d"}
    feat_cols = [c for c in df_cb_train.columns if c in df_cb_test.columns and c not in excluded]

    X_train_cb = df_cb_train.select(feat_cols).to_numpy().astype(np.float32)
    y_train_cb_gmv = df_cb_train["target"].to_numpy().astype(np.float32)
    was_act_train_cb = (df_cb_train["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_train_cb = (y_train_cb_gmv > 0).astype(int)

    X_test_cb = df_cb_test.select(feat_cols).to_numpy().astype(np.float32)
    was_act_test = (df_cb_test["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int) if "lifetime_gmv" in df_cb_test.columns else np.ones(n_users, dtype=int)

    print(f"[*] Training CB_REACT_FINAL on {len(df_cb_train):,} pooled rows...")
    m_react = was_act_train_cb == 0
    cb_react = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_react.fit(X_train_cb[m_react], will_buy_train_cb[m_react], verbose=False)
    cb_react_logits_test = cb_react.predict(X_test_cb, prediction_type="RawFormulaVal")

    print(f"[*] Training CB_CHURN_FINAL on {len(df_cb_train):,} pooled rows...")
    m_churn = was_act_train_cb == 1
    cb_churn = CatBoostClassifier(iterations=1500, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_churn.fit(X_train_cb[m_churn], (1 - will_buy_train_cb[m_churn]), verbose=False)
    cb_churn_logits_test = cb_churn.predict(X_test_cb, prediction_type="RawFormulaVal")

    print(f"[*] Training CB_AMOUNT_FINAL on {len(df_cb_train):,} pooled rows...")
    m_amt = y_train_cb_gmv > 0
    cb_amount = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
    cb_amount.fit(X_train_cb[m_amt], np.log1p(y_train_cb_gmv[m_amt]), verbose=False)
    cb_amount_z_test = np.maximum(0.0, cb_amount.predict(X_test_cb))

    del df_cb_train, X_train_cb, df_cb_test, X_test_cb
    gc.collect()

    # ------------------------------------------------------------------
    # Step B: Final Neural Specialists Training & Inference
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP B: TRAINING FINAL NEURAL SPECIALISTS ON ALL 250k USERS")
    print("=" * 80)

    train_users = pl.read_parquet(snap_dir / "snapshot_2026-01-14.parquet")["user_id"].to_numpy()
    n_train_users = len(train_users)

    max_ev = 180
    step_anchors = max(1, len(train_anchors) // 8)
    nn_train_anchors = [train_anchors[i] for i in range(0, len(train_anchors), step_anchors)][:8]
    if train_anchors[-1] not in nn_train_anchors:
        nn_train_anchors[-1] = train_anchors[-1]

    n_train_samples = len(nn_train_anchors) * n_train_users
    print(f"[*] Training Neural Specialists on {len(nn_train_anchors)} diverse anchors ({n_train_samples:,} samples): {nn_train_anchors}")

    mmap_dir = Path("scratch/mmap_run2")
    mmap_dir.mkdir(parents=True, exist_ok=True)

    train_c = np.memmap(mmap_dir / "train_c.dat", dtype=np.float16, mode="w+", shape=(n_train_samples, max_ev, 12))
    train_t = np.memmap(mmap_dir / "train_t.dat", dtype=np.float16, mode="w+", shape=(n_train_samples, max_ev, 12))
    train_r = np.memmap(mmap_dir / "train_r.dat", dtype=np.int16, mode="w+", shape=(n_train_samples, max_ev))
    train_m = np.memmap(mmap_dir / "train_m.dat", dtype=bool, mode="w+", shape=(n_train_samples, max_ev))
    train_m[:] = True
    train_emp = np.memmap(mmap_dir / "train_emp.dat", dtype=bool, mode="w+", shape=(n_train_samples,))
    train_emp[:] = False
    train_z = np.memmap(mmap_dir / "train_z.dat", dtype=np.float32, mode="w+", shape=(n_train_samples,))
    train_act = np.memmap(mmap_dir / "train_act.dat", dtype=np.float32, mode="w+", shape=(n_train_samples,))
    train_buy = np.memmap(mmap_dir / "train_buy.dat", dtype=np.float32, mode="w+", shape=(n_train_samples,))
    train_rub = np.memmap(mmap_dir / "train_rub.dat", dtype=np.float32, mode="w+", shape=(n_train_samples,))

    for a_idx, a_str in enumerate(nn_train_anchors):
        off = a_idx * n_train_users
        snap_a = pl.read_parquet(fs_dir / f"anchor_{a_str}.parquet")
        u_map = {u: i for i, u in enumerate(snap_a["user_id"].to_list())}
        order = [u_map[u] for u in train_users if u in u_map]
        y_r = snap_a["target"].to_numpy()[order].astype(np.float32)
        past_g = snap_a["lifetime_gmv"].to_numpy()[order].astype(np.float32)

        train_z[off : off + len(order)] = np.log1p(np.maximum(0.0, y_r))
        train_act[off : off + len(order)] = (past_g > 0).astype(np.float32)
        train_buy[off : off + len(order)] = (y_r > 0).astype(np.float32)
        train_rub[off : off + len(order)] = y_r

        extract_event_time_sequences(
            df_raw, train_users, a_str, max_events=max_ev,
            out_c=train_c, out_t=train_t, out_r=train_r, out_m=train_m, out_emp=train_emp,
            offset=off
        )
        print(f"   -> Extracted training sequences for {a_str} ({a_idx+1}/{len(nn_train_anchors)})")

    # Test anchor buffers for all 250k users
    test_c = np.memmap(mmap_dir / "test_c.dat", dtype=np.float16, mode="w+", shape=(n_users, max_ev, 12))
    test_t = np.memmap(mmap_dir / "test_t.dat", dtype=np.float16, mode="w+", shape=(n_users, max_ev, 12))
    test_r = np.memmap(mmap_dir / "test_r.dat", dtype=np.int16, mode="w+", shape=(n_users, max_ev))
    test_m = np.memmap(mmap_dir / "test_m.dat", dtype=bool, mode="w+", shape=(n_users, max_ev))
    test_m[:] = True
    test_emp = np.memmap(mmap_dir / "test_emp.dat", dtype=bool, mode="w+", shape=(n_users,))
    test_emp[:] = False
    dummy_z = np.zeros(n_users, dtype=np.float32)

    extract_event_time_sequences(
        df_raw, all_users, test_anchor, max_events=max_ev,
        out_c=test_c, out_t=test_t, out_r=test_r, out_m=test_m, out_emp=test_emp,
        offset=0
    )

    del df_raw
    gc.collect()

    train_ds = MemmapDataset(train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_buy, train_rub)
    test_ds = MemmapDataset(test_c, test_t, test_r, test_m, test_emp, dummy_z, was_act_test.astype(np.float32), dummy_z, dummy_z)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=torch.cuda.is_available())

    def train_and_infer_neural(model: nn.Module, n_steps: int = 4000, lr: float = 3e-4):
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        step = 0
        train_iter = iter(train_loader)

        while step < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            c, t, r, m, emp, z_t, act, buy, _ = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            optimizer.zero_grad()

            dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)

            p_react = torch.sigmoid(r_logit)
            p_churn = torch.sigmoid(c_logit)
            p_buy = torch.where(act > 0.5, 1.0 - p_churn, p_react)
            fact_z = p_buy * cond_z

            l_fact = F.mse_loss(fact_z, z_t)
            l_dir = F.mse_loss(dir_z, z_t)

            pos_mask = buy > 0.5
            l_cond = F.mse_loss(cond_z[pos_mask], z_t[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=device)

            inact_mask = act <= 0.5
            l_react = F.binary_cross_entropy_with_logits(r_logit[inact_mask], buy[inact_mask]) if inact_mask.sum() > 0 else torch.tensor(0.0, device=device)

            act_mask = act > 0.5
            l_churn = F.binary_cross_entropy_with_logits(c_logit[act_mask], 1.0 - buy[act_mask]) if act_mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = 1.00 * l_fact + 0.25 * l_dir + 0.25 * l_cond + 0.10 * l_react + 0.10 * l_churn
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

        model.eval()
        r_logits_list, c_logits_list, cond_z_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                c, t, r, m, emp, _, _, _, _ = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
                r_logits_list.append(r_logit.cpu().numpy())
                c_logits_list.append(c_logit.cpu().numpy())
                cond_z_list.append(cond_z.cpu().numpy())

        return (
            np.concatenate(r_logits_list),
            np.concatenate(c_logits_list),
            np.concatenate(cond_z_list),
        )

    # Train S1
    print("\n[*] Training Final S1 Masked GRU Specialists...")
    s1_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s1_r_test, s1_c_test, s1_cond_test = train_and_infer_neural(s1_model, n_steps=3000, lr=5e-4)

    # Train S2
    print("\n[*] Training Final S2 Dense GRU Specialists...")
    s2_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s2_r_test, s2_c_test, s2_cond_test = train_and_infer_neural(s2_model, n_steps=3000, lr=5e-4)

    # Train ETT
    print("\n[*] Training Final Event-Time Transformer Specialists (180 tok, tau=30d)...")
    ett_model = EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=max_ev)
    ett_r_test, ett_c_test, ett_cond_test = train_and_infer_neural(ett_model, n_steps=4500, lr=3e-4)

    # Cleanup mmap scratch files
    del train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_buy, train_rub
    del test_c, test_t, test_r, test_m, test_emp
    gc.collect()
    shutil.rmtree(mmap_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Step C: Assemble Final Submission using RUN 1 Meta-Weights
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP C: ASSEMBLING FINAL SUBMISSION WITH RUN 1 META-WEIGHTS")
    print("=" * 80)

    # Stacking
    X_r_test = np.column_stack([cb_react_logits_test, s1_r_test, s2_r_test, ett_r_test])
    p_react = expit(np.dot(X_r_test, w_react))

    X_c_test = np.column_stack([cb_churn_logits_test, s1_c_test, s2_c_test, ett_c_test])
    p_churn = expit(np.dot(X_c_test, w_churn))

    p_buy = np.where(was_act_test == 0, p_react, 1.0 - p_churn)

    X_amt_test = np.column_stack([cb_amount_z_test, s1_cond_test, s2_cond_test, ett_cond_test])
    conditional_z = np.maximum(0.0, np.dot(X_amt_test, ridge_coef) + ridge_intercept)

    z_prediction = np.power(p_buy, alpha) * conditional_z
    gmv_prediction = np.expm1(np.clip(z_prediction, 0.0, None))

    # Match exact sample_submit.csv ordering
    df_sub = pl.DataFrame({
        "user_id": all_users,
        "predict": gmv_prediction,
    })

    sub_path = Path("submission_specialized_hurdle_canonical_250k.csv")
    df_sub.write_csv(sub_path)
    df_sub.write_csv(Path("submission_specialized_hurdle_stack.csv"))

    print(f"\n[+] FINAL SUBMISSION SAVED TO {sub_path} ({len(df_sub):,} rows)")
    print(f"    Mean GMV: {np.mean(gmv_prediction):.2f} RUB")
    print(f"    Median GMV: {np.median(gmv_prediction):.2f} RUB")
    print(f"    Zero GMV Ratio (GMV == 0): {np.mean(gmv_prediction == 0.0) * 100:.2f}%")


if __name__ == "__main__":
    main()
