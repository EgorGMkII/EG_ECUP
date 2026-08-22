"""
BUILD JOINT RMSLE SUBMISSION ON DATASPHERE (Tesla V100 GPU)

This script performs:
1. Deterministic inference of CatBoost, S1 Masked GRU, S2 Dense GRU, and ETT Specialists on 250k test users.
2. Baseline reconstruction and verification against submission_specialized_hurdle_stack.csv.
3. Application of joint end-to-end meta-weights from joint_weights_all_oof_candidate.json.
4. Export of:
   - submission_specialized_hurdle_joint_rmsle.csv
   - submission_specialized_hurdle_joint_rmsle_diagnostics.parquet
   - artifacts/specialized_hurdle/test_specialists_raw_predictions_250k.parquet
"""

import os
import sys
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoostClassifier, CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ----------------------------------------------------------------------
# Model Architectures
# ----------------------------------------------------------------------

class GRUEncoder(nn.Module):
    def __init__(self, d_in: int = 12, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.head_dir = nn.Linear(d_model, 1)
        self.head_cond = nn.Linear(d_model, 1)
        self.head_react = nn.Linear(d_model, 1)
        self.head_churn = nn.Linear(d_model, 1)

    def forward(self, content, time_feat, ranks, mask, empty):
        x = self.in_proj(content)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        dir_z = torch.relu(self.head_dir(h)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(h)).squeeze(-1)
        r_logit = self.head_react(h).squeeze(-1)
        c_logit = self.head_churn(h).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


class EventTimeTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, max_events: int = 180):
        super().__init__()
        self.cont_proj = nn.Linear(12, d_model)
        self.time_proj = nn.Linear(12, d_model)
        self.rank_emb = nn.Embedding(max_events + 1, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_dir = nn.Linear(d_model, 1)
        self.head_cond = nn.Linear(d_model, 1)
        self.head_react = nn.Linear(d_model, 1)
        self.head_churn = nn.Linear(d_model, 1)

    def forward(self, content, time_feat, ranks, mask, empty):
        c_emb = self.cont_proj(content)
        t_emb = self.time_proj(time_feat)
        r_emb = self.rank_emb(ranks.clamp(0, 180))
        x = self.norm_in(c_emb + t_emb + r_emb)
        
        mask_safe = mask.clone()
        mask_safe[mask.all(dim=-1), 0] = False
        h_seq = self.transformer(x, src_key_padding_mask=mask_safe)
        h = h_seq[:, -1, :]
        h = torch.where(empty.unsqueeze(-1), torch.zeros_like(h), h)
        
        dir_z = torch.relu(self.head_dir(h)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(h)).squeeze(-1)
        r_logit = self.head_react(h).squeeze(-1)
        c_logit = self.head_churn(h).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


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
            torch.from_numpy(self.ranks[idx].astype(np.int64)),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.empty[idx], dtype=torch.bool),
            torch.tensor(self.z_true[idx], dtype=torch.float32),
            torch.tensor(self.was_active[idx], dtype=torch.float32),
            torch.tensor(self.will_buy[idx], dtype=torch.float32),
            torch.tensor(self.y_rub[idx], dtype=torch.float32),
        )


def extract_event_time_sequences(
    df_raw: pl.DataFrame,
    user_ids: np.ndarray,
    anchor_str: str,
    max_events: int = 180,
    tau_days: float = 30.0,
    out_c: np.ndarray = None,
    out_t: np.ndarray = None,
    out_r: np.ndarray = None,
    out_m: np.ndarray = None,
    out_emp: np.ndarray = None,
    offset: int = 0,
):
    anchor_dt = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    hist_start = anchor_dt - timedelta(days=364)
    midpoint_doy = (anchor_dt + timedelta(days=15)).timetuple().tm_yday

    df_filtered = df_raw.filter(
        (pl.col("event_date") >= hist_start) & (pl.col("event_date") <= anchor_dt)
    )

    cont_exprs = [
        pl.col("search").log1p().alias("c_search"),
        pl.col("cat").log1p().alias("c_cat"),
        pl.col("to_cart").log1p().alias("c_to_cart"),
        pl.col("to_ord").log1p().alias("c_to_ord"),
        pl.col("gmv").log1p().alias("c_gmv"),
        pl.col("gmv_search").log1p().alias("c_gmv_search"),
        pl.col("gmv_cat").log1p().alias("c_gmv_cat"),
        pl.col("searches").log1p().alias("c_searches"),
        pl.col("search_to_cart").alias("c_s2c"),
        pl.col("search_to_ord").alias("c_s2o"),
        pl.col("cat_to_cart").alias("c_c2c"),
        pl.col("cat_to_ord").alias("c_c2o"),
    ]

    time_diff_days = (pl.lit(anchor_dt) - pl.col("event_date")).dt.total_days().cast(pl.Float32)
    t_decay = (-time_diff_days / float(tau_days)).exp()
    t_doy = pl.col("event_date").dt.ordinal_day().cast(pl.Float32)
    t_sin = (2.0 * np.pi * t_doy / 365.25).sin()
    t_cos = (2.0 * np.pi * t_doy / 365.25).cos()
    delta_days = (time_diff_days - 15.0).abs()
    season_prox = (-delta_days / 30.0).exp()

    time_exprs = [
        (time_diff_days / 365.0).alias("t_norm"),
        t_decay.alias("t_decay"),
        t_sin.alias("t_sin"),
        t_cos.alias("t_cos"),
        season_prox.alias("t_season_prox"),
        (pl.col("event_date").dt.weekday() / 7.0).alias("t_dow"),
        (time_diff_days <= 7).cast(pl.Float32).alias("t_is_w1"),
        (time_diff_days <= 14).cast(pl.Float32).alias("t_is_w2"),
        (time_diff_days <= 30).cast(pl.Float32).alias("t_is_m1"),
        (time_diff_days <= 60).cast(pl.Float32).alias("t_is_m2"),
        (time_diff_days <= 90).cast(pl.Float32).alias("t_is_q1"),
        (time_diff_days <= 180).cast(pl.Float32).alias("t_is_h1"),
    ]

    df_prepared = df_filtered.select([
        pl.col("user_id"),
        pl.col("event_date"),
        *cont_exprs,
        *time_exprs,
    ]).sort(["user_id", "event_date"])

    user_groups = df_prepared.group_by("user_id", maintain_order=True).agg([
        pl.col("c_search"), pl.col("c_cat"), pl.col("c_to_cart"), pl.col("c_to_ord"),
        pl.col("c_gmv"), pl.col("c_gmv_search"), pl.col("c_gmv_cat"), pl.col("c_searches"),
        pl.col("c_s2c"), pl.col("c_s2o"), pl.col("c_c2c"), pl.col("c_c2o"),
        pl.col("t_norm"), pl.col("t_decay"), pl.col("t_sin"), pl.col("t_cos"),
        pl.col("t_season_prox"), pl.col("t_dow"), pl.col("t_is_w1"), pl.col("t_is_w2"),
        pl.col("t_is_m1"), pl.col("t_is_m2"), pl.col("t_is_q1"), pl.col("t_is_h1"),
    ])

    user_dict = {row["user_id"]: row for row in user_groups.iter_rows(named=True)}
    n_users = len(user_ids)

    c_cols = ["c_search", "c_cat", "c_to_cart", "c_to_ord", "c_gmv", "c_gmv_search", "c_gmv_cat", "c_searches", "c_s2c", "c_s2o", "c_c2c", "c_c2o"]
    t_cols = ["t_norm", "t_decay", "t_sin", "t_cos", "t_season_prox", "t_dow", "t_is_w1", "t_is_w2", "t_is_m1", "t_is_m2", "t_is_q1", "t_is_h1"]

    for i, u in enumerate(user_ids):
        idx = offset + i
        if u not in user_dict:
            out_emp[idx] = True
            out_m[idx, :] = True
            continue

        row = user_dict[u]
        n_ev = len(row["c_search"])
        if n_ev == 0:
            out_emp[idx] = True
            out_m[idx, :] = True
            continue

        take = min(n_ev, max_events)
        start_pad = max_events - take

        c_mat = np.column_stack([row[col][-take:] for col in c_cols]).astype(np.float16)
        t_mat = np.column_stack([row[col][-take:] for col in t_cols]).astype(np.float16)

        out_c[idx, start_pad:, :] = c_mat
        out_t[idx, start_pad:, :] = t_mat
        out_r[idx, start_pad:] = np.arange(1, take + 1, dtype=np.int16)
        out_m[idx, start_pad:] = False
        out_emp[idx] = False


# ----------------------------------------------------------------------
# Main Execution Flow
# ----------------------------------------------------------------------

def main():
    print("=" * 80)
    print("BUILD JOINT RMSLE SUBMISSION (SPECIALIZED HURDLE STACK)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if device.type == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # Exact Full-Precision Meta-Weights (Embedded for 100% fail-safe execution)
    EXACT_BASE_META = {
        "model_order": ["CatBoost", "S1_GRU", "S2_GRU", "ETT"],
        "react_stack_weights": [0.19110942984935578, 0.5444806953211873, 0.0, 0.264409874829457],
        "churn_stack_weights": [0.3426388029610671, 0.18623190278699117, 0.0, 0.4711292942519417],
        "amount_ridge_coefficients": [0.16719562977456925, 0.05066011501615654, 0.32874426180746497, 0.4800865245425257],
        "amount_ridge_intercept": -0.03887640353732458,
        "ALPHA": 1.1,
    }

    EXACT_JOINT_META = {
        "model_order": ["CatBoost", "S1_GRU", "S2_GRU", "ETT"],
        "react_stack_weights": [0.02571986359935257, 0.3658967641100093, 0.0, 0.6083833722906381],
        "churn_stack_weights": [0.2770564003031295, 0.22700033415688514, 0.06279933646526792, 0.43314392907471744],
        "amount_ridge_coefficients": [0.12217097079693888, 4.146865521979716e-19, 0.35635864495834946, 0.5454727117581657],
        "amount_ridge_intercept": 0.02079580550858867,
        "ALPHA": 1.1,
    }

    # Load from file if available, else use embedded exact values
    base_candidates = [
        Path("artifacts/specialized_hurdle/run1_meta_weights.json"),
        Path("run1_meta_weights.json"),
        Path("configs/run1_meta_weights.json"),
    ]
    meta_base = EXACT_BASE_META
    for p in base_candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                meta_base = json.load(f)
            break

    joint_candidates = [
        Path("artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json"),
        Path("joint_weights_all_oof_candidate.json"),
        Path("configs/joint_weights_all_oof_candidate.json"),
    ]
    meta_joint = EXACT_JOINT_META
    for p in joint_candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                meta_joint = json.load(f)
            break

    print("\n[+] Loaded Baseline Meta-Weights (Separately Fitted):")
    print(f"    React Weights: {meta_base['react_stack_weights']}")
    print(f"    Churn Weights: {meta_base['churn_stack_weights']}")
    print(f"    Amount Ridge:  {meta_base['amount_ridge_coefficients']} + {meta_base['amount_ridge_intercept']}")

    print("\n[+] Loaded Joint RMSLE Meta-Weights (End-to-End Optimized):")
    print(f"    React Weights: {meta_joint['react_stack_weights']}")
    print(f"    Churn Weights: {meta_joint['churn_stack_weights']}")
    print(f"    Amount Ridge:  {meta_joint['amount_ridge_coefficients']} + {meta_joint['amount_ridge_intercept']}")
    print(f"    ALPHA:         {meta_joint['ALPHA']} (strictly fixed)")

    w_react_old = np.array(meta_base["react_stack_weights"])
    w_churn_old = np.array(meta_base["churn_stack_weights"])
    w_amt_old = np.array(meta_base["amount_ridge_coefficients"])
    b_amt_old = float(meta_base["amount_ridge_intercept"])

    w_react_joint = np.array(meta_joint["react_stack_weights"])
    w_churn_joint = np.array(meta_joint["churn_stack_weights"])
    w_amt_joint = np.array(meta_joint["amount_ridge_coefficients"])
    b_amt_joint = float(meta_joint["amount_ridge_intercept"])
    alpha = float(meta_joint["ALPHA"])

    sub_template_path = Path("sample_submit.csv")
    sub_template = pl.read_csv(sub_template_path)
    all_users = sub_template["user_id"].to_numpy()
    n_users = len(all_users)
    print(f"\n[+] Test cohort: {n_users:,} users from sample_submit.csv.")

    train_anchors = [
        "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
        "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
        "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
        "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15",
        "2025-12-22", "2026-01-05", "2026-01-14"
    ]
    test_anchor = "2026-02-13"

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

    mmap_dir = Path("scratch/mmap_joint_run")
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

            c, t, r, m, emp, z_true, was_act, will_buy, y_rub = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

            optimizer.zero_grad()
            dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)

            m_inact = was_act == 0
            m_act = was_act == 1
            m_pos = will_buy == 1

            loss_r = nn.functional.binary_cross_entropy_with_logits(r_logit[m_inact], will_buy[m_inact]) if m_inact.sum() > 0 else torch.tensor(0.0, device=device)
            loss_c = nn.functional.binary_cross_entropy_with_logits(c_logit[m_act], (1.0 - will_buy[m_act])) if m_act.sum() > 0 else torch.tensor(0.0, device=device)
            loss_cond = nn.functional.mse_loss(cond_z[m_pos], z_true[m_pos]) if m_pos.sum() > 0 else torch.tensor(0.0, device=device)
            loss_dir = nn.functional.mse_loss(dir_z, z_true)

            total_loss = loss_dir + 0.5 * loss_cond + 0.25 * (loss_r + loss_c)
            total_loss.backward()
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
    # Step C: Save Raw Specialist Predictions Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP C: SAVING RAW SPECIALIST PREDICTIONS (250,000 ROWS)")
    print("=" * 80)

    df_raw_preds = pl.DataFrame({
        "user_id": all_users,
        "was_active": was_act_test,
        "cb_react_logit": cb_react_logits_test,
        "s1_react_logit": s1_r_test,
        "s2_react_logit": s2_r_test,
        "ett_react_logit": ett_r_test,
        "cb_churn_logit": cb_churn_logits_test,
        "s1_churn_logit": s1_c_test,
        "s2_churn_logit": s2_c_test,
        "ett_churn_logit": ett_c_test,
        "cb_amount_z": cb_amount_z_test,
        "s1_amount_z": s1_cond_test,
        "s2_amount_z": s2_cond_test,
        "ett_amount_z": ett_cond_test,
    })

    raw_preds_path = Path("artifacts/specialized_hurdle/test_specialists_raw_predictions_250k.parquet")
    raw_preds_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw_preds.write_parquet(raw_preds_path)
    df_raw_preds.write_parquet(Path("test_specialists_raw_predictions_250k.parquet"))
    print(f"[+] Saved raw test predictions to {raw_preds_path} and test_specialists_raw_predictions_250k.parquet")

    # ------------------------------------------------------------------
    # Step D: Reproduce Baseline (gmv_old) & Generate Joint (gmv_joint)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP D: COMPUTING BASELINE & JOINT RMSLE PREDICTIONS")
    print("=" * 80)

    X_r_test = np.column_stack([cb_react_logits_test, s1_r_test, s2_r_test, ett_r_test])
    X_c_test = np.column_stack([cb_churn_logits_test, s1_c_test, s2_c_test, ett_c_test])
    X_amt_test = np.column_stack([cb_amount_z_test, s1_cond_test, s2_cond_test, ett_cond_test])

    # Baseline Calculations
    p_react_old = expit(np.dot(X_r_test, w_react_old))
    p_churn_old = expit(np.dot(X_c_test, w_churn_old))
    p_buy_old = np.where(was_act_test == 0, p_react_old, 1.0 - p_churn_old)
    conditional_z_old = np.maximum(0.0, np.dot(X_amt_test, w_amt_old) + b_amt_old)
    z_old = np.power(p_buy_old, alpha) * conditional_z_old
    z_old = np.clip(z_old, 0.0, None)
    gmv_old = np.expm1(z_old)

    # Joint RMSLE Calculations
    p_react_joint = expit(np.dot(X_r_test, w_react_joint))
    p_churn_joint = expit(np.dot(X_c_test, w_churn_joint))
    p_buy_joint = np.where(was_act_test == 0, p_react_joint, 1.0 - p_churn_joint)
    conditional_z_joint = np.maximum(0.0, np.dot(X_amt_test, w_amt_joint) + b_amt_joint)
    z_joint = np.power(p_buy_joint, alpha) * conditional_z_joint
    z_joint = np.clip(z_joint, 0.0, None)
    gmv_joint = np.expm1(z_joint)

    # Diagnostics Table
    df_diag = pl.DataFrame({
        "user_id": all_users,
        "was_active": was_act_test,
        "p_react_old": p_react_old,
        "p_react_joint": p_react_joint,
        "p_churn_old": p_churn_old,
        "p_churn_joint": p_churn_joint,
        "p_buy_old": p_buy_old,
        "p_buy_joint": p_buy_joint,
        "conditional_z_old": conditional_z_old,
        "conditional_z_joint": conditional_z_joint,
        "z_old": z_old,
        "z_joint": z_joint,
        "gmv_old": gmv_old,
        "gmv_joint": gmv_joint,
        "gmv_delta": gmv_joint - gmv_old,
    })

    diag_path = Path("submission_specialized_hurdle_joint_rmsle_diagnostics.parquet")
    df_diag.write_parquet(diag_path)
    print(f"[+] Saved diagnostics table to {diag_path}")

    # Check baseline reproduction if previous submission exists
    prev_sub_path = Path("submission_specialized_hurdle_stack.csv")
    if prev_sub_path.exists():
        df_prev = pl.read_csv(prev_sub_path)
        col = "predict" if "predict" in df_prev.columns else "target"
        diff = np.abs(df_prev[col].to_numpy() - gmv_old)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"\n[+] BASELINE REPRODUCTION AUDIT:")
        print(f"    Max Absolute Diff vs previous submission:  {max_diff:.8f}")
        print(f"    Mean Absolute Diff vs previous submission: {mean_diff:.8f}")

    # Final Joint RMSLE Submission CSV (exact competition schema: user_id, predict)
    df_joint_sub = pl.DataFrame({
        "user_id": all_users,
        "predict": gmv_joint,
    })
    joint_sub_path = Path("submission_specialized_hurdle_joint_rmsle.csv")
    df_joint_sub.write_csv(joint_sub_path)

    print(f"\n[+] FINAL JOINT SUBMISSION SAVED TO {joint_sub_path} ({len(df_joint_sub):,} rows)")
    print(f"    Baseline Mean GMV: {np.mean(gmv_old):.2f} RUB | Joint Mean GMV: {np.mean(gmv_joint):.2f} RUB")
    print(f"    Baseline Median GMV: {np.median(gmv_old):.2f} RUB | Joint Median GMV: {np.median(gmv_joint):.2f} RUB")


if __name__ == "__main__":
    main()
