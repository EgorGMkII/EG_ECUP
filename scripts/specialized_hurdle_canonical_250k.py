"""CANONICAL SPECIALIZED HURDLE STACK (RUN 1 ON 100K + RUN 2 ON 250K TEST INFERENCE)

Reproduces EXACTLY the 100k winning pipeline (LB 1.664077 / 1.6649):
1. RUN 1 (Meta-Weights Fitting):
   - Cohort: 100,000 canonical users (selected_users_100k.parquet).
   - Training Anchors: 17 anchors (2025-03-31 .. 2025-11-24) on data/snapshots/.
   - Meta-Anchor: 2025-12-15.
   - Specialists: CatBoost (3 models), S1 Masked GRU, S2 Dense GRU, Event-Time Transformer.
   - Meta-Weights Fitting:
     * React Stack: Simplex LogLoss (sum(w)=1.0, w>=0) on inactive users.
     * Churn Stack: Simplex LogLoss (sum(w)=1.0, w>=0) on active users.
     * Amount Stack: Ridge(alpha=1.0, positive=True, fit_intercept=True) on positive buyers.
     * Alpha = 1.1 Hurdle exponent.
2. RUN 2 (Final Specialist Training & 250k Test Inference):
   - Training: All 23 anchors (2025-03-31 .. 2026-01-14, 2.3M rows / 800k sequences).
   - Test Inference: Full 250,000 users on anchor 2026-02-13.
   - Applies fitted RUN 1 meta-weights to produce final submission.
"""

import gc
import json
import math
import os
import sys
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

CANONICAL_ANCHORS = [
    "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
    "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
    "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
    "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15",
    "2025-12-22", "2026-01-05", "2026-01-14"
]

# ----------------------------------------------------------------------
# Neural Architectures (Canonical)
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
        
        h = self.transformer(x, src_key_padding_mask=mask_safe)
        
        valid_mask = (~mask_safe).unsqueeze(-1).float()
        pooled = (h * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        
        dir_z = torch.relu(self.head_dir(pooled)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(pooled)).squeeze(-1)
        r_logit = self.head_react(pooled).squeeze(-1)
        c_logit = self.head_churn(pooled).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


class SequenceDataset(Dataset):
    def __init__(self, content, time_feat, ranks, mask, empty, z_true=None, was_active=None, will_buy=None):
        self.content = content
        self.time_feat = time_feat
        self.ranks = ranks
        self.mask = mask
        self.empty = empty
        self.z_true = z_true
        self.was_active = was_active
        self.will_buy = will_buy

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        item = [
            torch.from_numpy(self.content[idx]),
            torch.from_numpy(self.time_feat[idx]),
            torch.from_numpy(self.ranks[idx].astype(np.int64)),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(bool(self.empty[idx]), dtype=torch.bool),
        ]
        if self.z_true is not None:
            item.extend([
                torch.tensor(float(self.z_true[idx]), dtype=torch.float32),
                torch.tensor(float(self.was_active[idx]), dtype=torch.float32),
                torch.tensor(float(self.will_buy[idx]), dtype=torch.float32),
            ])
        return tuple(item)


# ----------------------------------------------------------------------
# Fast Sequence Extraction
# ----------------------------------------------------------------------

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

    df_daily = (
        df_filtered.group_by(["user_id", "event_date"])
        .agg([
            pl.col("search").sum().alias("search"),
            pl.col("cat").sum().alias("cat"),
            pl.col("has_search_to_cart").max().alias("has_search_to_cart"),
            pl.col("has_search_to_ord").max().alias("has_search_to_ord"),
            pl.col("has_cat_to_cart").max().alias("has_cat_to_cart"),
            pl.col("has_cat_to_ord").max().alias("has_cat_to_ord"),
            pl.col("search_to_cart").sum().alias("search_to_cart"),
            pl.col("search_to_ord").sum().alias("search_to_ord"),
            pl.col("cat_to_cart").sum().alias("cat_to_cart"),
            pl.col("cat_to_ord").sum().alias("cat_to_ord"),
            pl.col("gmv").sum().alias("gmv"),
            pl.col("to_ord").sum().alias("to_ord"),
        ])
        .sort(["user_id", "event_date"])
    )

    df_daily = df_daily.with_columns([
        pl.col("search").cast(pl.Float32).log1p().alias("c0_search"),
        pl.col("cat").cast(pl.Float32).log1p().alias("c1_cat"),
        pl.col("has_search_to_cart").cast(pl.Float32).alias("c2_has_search_to_cart"),
        pl.col("has_search_to_ord").cast(pl.Float32).alias("c3_has_search_to_ord"),
        pl.col("has_cat_to_cart").cast(pl.Float32).alias("c4_has_cat_to_cart"),
        pl.col("has_cat_to_ord").cast(pl.Float32).alias("c5_has_cat_to_ord"),
        pl.col("search_to_cart").log1p().alias("c6_log_search_to_cart"),
        pl.col("search_to_ord").log1p().alias("c7_log_search_to_ord"),
        pl.col("cat_to_cart").log1p().alias("c8_log_cat_to_cart"),
        pl.col("cat_to_ord").log1p().alias("c9_log_cat_to_ord"),
        pl.col("gmv").log1p().alias("c10_log_gmv"),
        (pl.col("to_ord") > 0).cast(pl.Float32).alias("c11_is_purchase_day"),
    ])

    c_cols = [f"c{k}_{name}" for k, name in enumerate([
        "search", "cat", "has_search_to_cart", "has_search_to_ord",
        "has_cat_to_cart", "has_cat_to_ord", "log_search_to_cart",
        "log_search_to_ord", "log_cat_to_cart", "log_cat_to_ord",
        "log_gmv", "is_purchase_day"
    ])]

    user_to_events = {}
    for row in df_daily.iter_rows(named=True):
        u = row["user_id"]
        if u not in user_to_events:
            user_to_events[u] = []
        user_to_events[u].append(row)

    del df_filtered, df_daily
    gc.collect()

    for i, uid in enumerate(user_ids):
        idx = offset + i
        evs = user_to_events.get(uid, [])
        if len(evs) == 0:
            out_emp[idx] = True
            out_m[idx, 0] = False
            out_r[idx, 0] = 0
            continue

        evs = evs[-max_events:]
        num_ev = len(evs)

        prev_date = None
        for j, ev in enumerate(evs):
            pos = max_events - num_ev + j
            out_m[idx, pos] = False
            rank_from_end = num_ev - 1 - j
            out_r[idx, pos] = rank_from_end

            out_c[idx, pos] = [ev[col] for col in c_cols]

            ev_date = ev["event_date"]
            age_days = float((anchor_dt - ev_date).days)
            age_days = max(0.0, min(365.0, age_days))

            if j == 0:
                is_first_event = 1.0
                delta_days = 0.0
            else:
                is_first_event = 0.0
                delta_days = float((ev_date - prev_date).days)
                delta_days = max(0.0, min(365.0, delta_days))
            prev_date = ev_date

            t_norm = age_days / 365.0
            t_decay = math.exp(-age_days / tau_days)
            log_delta = math.log1p(delta_days)

            doy = ev_date.timetuple().tm_yday
            sin_doy = math.sin(2.0 * math.pi * doy / 365.25)
            cos_doy = math.cos(2.0 * math.pi * doy / 365.25)

            delta_target = abs(doy - midpoint_doy)
            delta_target = min(delta_target, 365 - delta_target)
            t_to_mid = delta_target / 182.5

            dow = ev_date.weekday()
            sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
            cos_dow = math.cos(2.0 * math.pi * dow / 7.0)
            is_weekend = 1.0 if dow in [5, 6] else 0.0

            out_t[idx, pos] = [
                t_norm, t_decay, log_delta,
                float(rank_from_end) / float(max_events),
                is_first_event, 1.0 - is_first_event,
                sin_doy, cos_doy, t_to_mid,
                sin_dow, cos_dow, is_weekend
            ]


# ----------------------------------------------------------------------
# MAIN RUN 1 + RUN 2 PIPELINE
# ----------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CANONICAL SPECIALIZED HURDLE PIPELINE (RUN 1: 100K -> RUN 2: 250K)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path("artifacts/specialized_hurdle_canonical_250k")
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_dir = out_dir / "feature_store"
    fs_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    df_raw = pl.read_parquet(raw_path)

    # 1. Cohort Users for RUN 1 (100k Canonical Cohort)
    users_100k_path = Path("selected_users_100k.parquet")
    if not users_100k_path.exists():
        users_100k_path = Path("data/selected_users_100k.parquet")
    if not users_100k_path.exists():
        users_100k_path = Path("artifacts/selected_users_100k.parquet")

    users_100k = pl.read_parquet(users_100k_path)["user_id"].to_numpy()
    n_users_r1 = len(users_100k)
    print(f"[+] Loaded {n_users_r1:,} canonical 100k cohort users for RUN 1.")

    all_users = df_raw.select(pl.col("user_id").unique().sort())["user_id"].to_numpy()
    n_all_users = len(all_users)
    print(f"[+] Loaded {n_all_users:,} total cohort users for RUN 2 test inference.")

    # 2. RUN 1: Training Anchors and Meta-Anchor
    all_anchors = CANONICAL_ANCHORS
    cutoff = pd.Timestamp("2025-12-15")
    train_anchors_r1 = []
    for a in all_anchors:
        a_dt = pd.Timestamp(a)
        t_end = a_dt + pd.Timedelta(days=30)
        if t_end <= cutoff:
            train_anchors_r1.append(a)

    meta_anchor = "2025-12-15"
    print(f"\n[+] RUN 1: {len(train_anchors_r1)} training anchors: {train_anchors_r1[0]} .. {train_anchors_r1[-1]}")
    print(f"[+] RUN 1: Meta-Anchor for Stack Fitting: {meta_anchor}")

    # Step A: Build Feature Store for RUN 1 on VM
    from src.specialized_hurdle.feature_store import build_causal_feature_store
    snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")
    needed_anchors = train_anchors_r1 + [meta_anchor]
    missing = [a for a in needed_anchors if not (fs_dir / f"anchor_{a}.parquet").exists()]
    if missing:
        print(f"[*] Building causal feature store for {len(missing)} anchors on 100k users...")
        build_causal_feature_store(
            snapshots_dir=snap_dir,
            out_dir=fs_dir,
            anchors=needed_anchors,
            user_ids=users_100k,
        )

    train_dfs_r1 = [pl.read_parquet(fs_dir / f"anchor_{a}.parquet") for a in train_anchors_r1]
    df_cb_tr1 = pl.concat(train_dfs_r1)
    df_cb_meta = pl.read_parquet(fs_dir / f"anchor_{meta_anchor}.parquet")

    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d"}
    feat_cols = [c for c in df_cb_tr1.columns if c not in excluded]

    X_tr1_cb = df_cb_tr1.select(feat_cols).to_numpy().astype(np.float32)
    y_tr1_cb_gmv = df_cb_tr1["target"].to_numpy().astype(np.float32)
    was_act_tr1_cb = (df_cb_tr1["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_tr1_cb = (y_tr1_cb_gmv > 0).astype(int)

    X_meta_cb = df_cb_meta.select(feat_cols).to_numpy().astype(np.float32)
    y_meta_cb_gmv = df_cb_meta["target"].to_numpy().astype(np.float32)
    was_act_meta_cb = (df_cb_meta["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_meta_cb = (y_meta_cb_gmv > 0).astype(int)

    print(f"\n[*] RUN 1: Training CatBoost Specialists on {len(df_cb_tr1):,} rows...")
    m_react_tr = was_act_tr1_cb == 0
    m_react_va = was_act_meta_cb == 0
    cb_react = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_react.fit(X_tr1_cb[m_react_tr], will_buy_tr1_cb[m_react_tr], eval_set=(X_meta_cb[m_react_va], will_buy_meta_cb[m_react_va]), early_stopping_rounds=80, verbose=False)
    cb_react_logits_meta = cb_react.predict(X_meta_cb, prediction_type="RawFormulaVal")

    m_churn_tr = was_act_tr1_cb == 1
    m_churn_va = was_act_meta_cb == 1
    cb_churn = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_churn.fit(X_tr1_cb[m_churn_tr], (1 - will_buy_tr1_cb[m_churn_tr]), eval_set=(X_meta_cb[m_churn_va], (1 - will_buy_meta_cb[m_churn_va])), early_stopping_rounds=80, verbose=False)
    cb_churn_logits_meta = cb_churn.predict(X_meta_cb, prediction_type="RawFormulaVal")

    m_amt_tr = y_tr1_cb_gmv > 0
    m_amt_va = y_meta_cb_gmv > 0
    cb_amount = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
    cb_amount.fit(X_tr1_cb[m_amt_tr], np.log1p(y_tr1_cb_gmv[m_amt_tr]), eval_set=(X_meta_cb[m_amt_va], np.log1p(y_meta_cb_gmv[m_amt_va])), early_stopping_rounds=80, verbose=False)
    cb_amount_z_meta = np.maximum(0.0, cb_amount.predict(X_meta_cb))

    del df_cb_tr1, X_tr1_cb
    gc.collect()

    # Step B: Sequential Sequences & Neural Training for RUN 1
    print("\n[*] RUN 1: Extracting Neural Sequences (8 sample anchors + meta on 100k users)...")
    sample_neural_anchors = ["2025-03-31", "2025-04-28", "2025-05-26", "2025-06-23", "2025-07-21", "2025-08-18", "2025-09-15", "2025-10-27"]
    max_ev = 180
    n_neural_tr = len(sample_neural_anchors) * n_users_r1

    tr_c = np.zeros((n_neural_tr, max_ev, 12), dtype=np.float32)
    tr_t = np.zeros((n_neural_tr, max_ev, 12), dtype=np.float32)
    tr_r = np.zeros((n_neural_tr, max_ev), dtype=np.int16)
    tr_m = np.ones((n_neural_tr, max_ev), dtype=bool)
    tr_emp = np.zeros(n_neural_tr, dtype=bool)

    tr_z = np.zeros(n_neural_tr, dtype=np.float32)
    tr_act = np.zeros(n_neural_tr, dtype=np.float32)
    tr_buy = np.zeros(n_neural_tr, dtype=np.float32)

    for k, a in enumerate(sample_neural_anchors):
        off = k * n_users_r1
        extract_event_time_sequences(df_raw, users_100k, a, max_ev, 30.0, tr_c, tr_t, tr_r, tr_m, tr_emp, off)
        df_a = pl.read_parquet(fs_dir / f"anchor_{a}.parquet")
        tr_z[off:off + n_users_r1] = np.log1p(df_a["target"].to_numpy().astype(np.float32))
        tr_act[off:off + n_users_r1] = (df_a["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(np.float32)
        tr_buy[off:off + n_users_r1] = (df_a["target"].to_numpy().astype(np.float32) > 0).astype(np.float32)

    meta_c = np.zeros((n_users_r1, max_ev, 12), dtype=np.float32)
    meta_t = np.zeros((n_users_r1, max_ev, 12), dtype=np.float32)
    meta_r = np.zeros((n_users_r1, max_ev), dtype=np.int16)
    meta_m = np.ones((n_users_r1, max_ev), dtype=bool)
    meta_emp = np.zeros(n_users_r1, dtype=bool)
    extract_event_time_sequences(df_raw, users_100k, meta_anchor, max_ev, 30.0, meta_c, meta_t, meta_r, meta_m, meta_emp, 0)

    train_ds = SequenceDataset(tr_c, tr_t, tr_r, tr_m, tr_emp, tr_z, tr_act, tr_buy)
    meta_ds = SequenceDataset(meta_c, meta_t, meta_r, meta_m, meta_emp)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    meta_loader = DataLoader(meta_ds, batch_size=512, shuffle=False)

    def train_neural_model(model: nn.Module, n_steps: int = 2500, lr: float = 5e-4):
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        bce_fn = nn.BCEWithLogitsLoss()
        mse_fn = nn.MSELoss()
        step = 0
        model.train()
        train_iter = iter(train_loader)
        while step < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            c, t, r, m, emp, z_true, was_act, will_buy = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            opt.zero_grad()
            dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
            loss_r = bce_fn(r_logit[was_act == 0], will_buy[was_act == 0]) if (was_act == 0).sum() > 0 else torch.tensor(0.0, device=device)
            loss_c = bce_fn(c_logit[was_act == 1], (1.0 - will_buy[was_act == 1])) if (was_act == 1).sum() > 0 else torch.tensor(0.0, device=device)
            loss_cond = mse_fn(cond_z[will_buy == 1], z_true[will_buy == 1]) if (will_buy == 1).sum() > 0 else torch.tensor(0.0, device=device)
            total_loss = loss_r + loss_c + loss_cond
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        model.eval()
        r_logits, c_logits, cond_zs = [], [], []
        with torch.no_grad():
            for batch in meta_loader:
                c, t, r, m, emp = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
                r_logits.append(r_logit.cpu().numpy())
                c_logits.append(c_logit.cpu().numpy())
                cond_zs.append(cond_z.cpu().numpy())
        return np.concatenate(r_logits), np.concatenate(c_logits), np.concatenate(cond_zs)

    print("\n[*] RUN 1: Training S1 Masked GRU Specialists...")
    s1_r, s1_c, s1_cond = train_neural_model(GRUEncoder(d_in=12, d_model=128, n_layers=2), n_steps=2500, lr=5e-4)

    print("\n[*] RUN 1: Training S2 Dense GRU Specialists...")
    s2_r, s2_c, s2_cond = train_neural_model(GRUEncoder(d_in=12, d_model=128, n_layers=2), n_steps=2500, lr=5e-4)

    print("\n[*] RUN 1: Training Event-Time Transformer Specialists...")
    ett_r, ett_c, ett_cond = train_neural_model(EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=max_ev), n_steps=3500, lr=3e-4)

    del tr_c, tr_t, tr_r, tr_m, tr_emp, tr_z, tr_act, tr_buy, meta_c, meta_t, meta_r, meta_m, meta_emp
    gc.collect()

    # Step C: Fit Canonical Meta-Weights
    print("\n" + "=" * 80)
    print("STEP C: FITTING CANONICAL META-WEIGHTS (SIMPLEX LOGLOSS & POSITIVE RIDGE)")
    print("=" * 80)

    inact_mask = was_act_meta_cb == 0
    act_mask = was_act_meta_cb == 1
    pos_mask = y_meta_cb_gmv > 0

    X_react = np.column_stack([cb_react_logits_meta[inact_mask], s1_r[inact_mask], s2_r[inact_mask], ett_r[inact_mask]])
    y_react = will_buy_meta_cb[inact_mask]

    def react_loss(w):
        w_norm = np.maximum(0.0, w) / max(1e-6, np.sum(np.maximum(0.0, w)))
        p = np.clip(expit(np.dot(X_react, w_norm)), 1e-6, 1.0 - 1e-6)
        return -np.mean(y_react * np.log(p) + (1.0 - y_react) * np.log(1.0 - p))

    res_react = minimize(react_loss, [0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)] * 4)
    w_react = np.maximum(0.0, res_react.x) / np.sum(np.maximum(0.0, res_react.x))
    print(f"[+] React Stack Weights (CB, S1, S2, ETT): {w_react.round(4).tolist()}")

    X_churn = np.column_stack([cb_churn_logits_meta[act_mask], s1_c[act_mask], s2_c[act_mask], ett_c[act_mask]])
    y_churn = (1 - will_buy_meta_cb[act_mask])

    def churn_loss(w):
        w_norm = np.maximum(0.0, w) / max(1e-6, np.sum(np.maximum(0.0, w)))
        p = np.clip(expit(np.dot(X_churn, w_norm)), 1e-6, 1.0 - 1e-6)
        return -np.mean(y_churn * np.log(p) + (1.0 - y_churn) * np.log(1.0 - p))

    res_churn = minimize(churn_loss, [0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)] * 4)
    w_churn = np.maximum(0.0, res_churn.x) / np.sum(np.maximum(0.0, res_churn.x))
    print(f"[+] Churn Stack Weights (CB, S1, S2, ETT): {w_churn.round(4).tolist()}")

    X_amount = np.column_stack([cb_amount_z_meta[pos_mask], s1_cond[pos_mask], s2_cond[pos_mask], ett_cond[pos_mask]])
    y_amount = np.log1p(y_meta_cb_gmv[pos_mask])

    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    ridge.fit(X_amount, y_amount)
    print(f"[+] Amount Ridge Coefficients: {ridge.coef_.round(4).tolist()} | Intercept: {ridge.intercept_:.4f}")

    meta_weights = {
        "model_order": ["CatBoost", "S1_GRU", "S2_GRU", "ETT"],
        "react_stack_weights": w_react.tolist(),
        "churn_stack_weights": w_churn.tolist(),
        "amount_ridge_coefficients": ridge.coef_.tolist(),
        "amount_ridge_intercept": float(ridge.intercept_),
        "ALPHA": 1.1,
        "meta_anchor": meta_anchor,
        "train_anchors_count": len(train_anchors_r1),
        "created_at": datetime.now().isoformat(),
    }

    with open(out_dir / "canonical_250k_meta_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta_weights, f, indent=2)
    with open("canonical_250k_meta_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta_weights, f, indent=2)

    # ------------------------------------------------------------------
    # RUN 2: TRAIN FINAL SPECIALISTS ON ALL 23 ANCHORS & 250K TEST INFERENCE
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUN 2: FINAL SPECIALIST TRAINING ON ALL 23 ANCHORS & 250K TEST INFERENCE")
    print("=" * 80)

    # Final feature store on all 23 anchors for 100k training pool
    all_23_anchors = CANONICAL_ANCHORS
    missing_23 = [a for a in all_23_anchors if not (fs_dir / f"anchor_{a}.parquet").exists()]
    if missing_23:
        print(f"[*] Building causal feature store for remaining {len(missing_23)} anchors...")
        build_causal_feature_store(
            snapshots_dir=snap_dir,
            out_dir=fs_dir,
            anchors=missing_23,
            user_ids=users_100k,
        )

    train_dfs_r2 = [pl.read_parquet(fs_dir / f"anchor_{a}.parquet") for a in all_23_anchors]
    df_cb_tr2 = pl.concat(train_dfs_r2)

    X_tr2_cb = df_cb_tr2.select(feat_cols).to_numpy().astype(np.float32)
    y_tr2_cb_gmv = df_cb_tr2["target"].to_numpy().astype(np.float32)
    was_act_tr2_cb = (df_cb_tr2["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_tr2_cb = (y_tr2_cb_gmv > 0).astype(int)

    # Build test snapshot for 2026-02-13 on ALL 250k users
    from src.snapshots import build_snapshot
    test_anchor = "2026-02-13"
    test_dt = pd.Timestamp(test_anchor).date()
    test_snap_path = out_dir / "snapshot_2026-02-13.parquet"
    if not test_snap_path.exists():
        print("[*] Building test snapshot for 2026-02-13 on ALL 250k users...")
        df_test_fs = build_snapshot(
            data=df_raw,
            user_ids=all_users.tolist(),
            anchor_date=test_dt,
            is_test=True,
        )
        df_test_fs.write_parquet(test_snap_path)
    else:
        df_test_fs = pl.read_parquet(test_snap_path)

    feat_cols = [c for c in df_cb_tr2.columns if c in df_test_fs.columns and c not in excluded]
    X_test_cb = df_test_fs.select(feat_cols).to_numpy().astype(np.float32)
    was_act_test = (df_test_fs["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int) if "lifetime_gmv" in df_test_fs.columns else np.ones(n_all_users, dtype=int)

    print(f"\n[*] RUN 2: Training FINAL CatBoost Specialists on {len(df_cb_tr2):,} rows...")
    m_r2_tr = was_act_tr2_cb == 0
    cb_react_final = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_react_final.fit(X_tr2_cb[m_r2_tr], will_buy_tr2_cb[m_r2_tr], verbose=False)
    cb_r_test = cb_react_final.predict(X_test_cb, prediction_type="RawFormulaVal")

    m_c2_tr = was_act_tr2_cb == 1
    cb_churn_final = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_churn_final.fit(X_tr2_cb[m_c2_tr], (1 - will_buy_tr2_cb[m_c2_tr]), verbose=False)
    cb_c_test = cb_churn_final.predict(X_test_cb, prediction_type="RawFormulaVal")

    m_a2_tr = y_tr2_cb_gmv > 0
    cb_amount_final = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
    cb_amount_final.fit(X_tr2_cb[m_a2_tr], np.log1p(y_tr2_cb_gmv[m_a2_tr]), verbose=False)
    cb_a_test = np.maximum(0.0, cb_amount_final.predict(X_test_cb))

    del df_cb_tr2, X_tr2_cb
    gc.collect()

    # Final Neural Test Sequence Extraction on all 250k users
    test_c = np.zeros((n_all_users, max_ev, 12), dtype=np.float32)
    test_t = np.zeros((n_all_users, max_ev, 12), dtype=np.float32)
    test_r = np.zeros((n_all_users, max_ev), dtype=np.int16)
    test_m = np.ones((n_all_users, max_ev), dtype=bool)
    test_emp = np.zeros(n_all_users, dtype=bool)
    extract_event_time_sequences(df_raw, all_users, test_anchor, max_ev, 30.0, test_c, test_t, test_r, test_m, test_emp, 0)
    test_ds = SequenceDataset(test_c, test_t, test_r, test_m, test_emp)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    def train_final_neural(model: nn.Module, n_steps: int = 2500, lr: float = 5e-4):
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        bce_fn = nn.BCEWithLogitsLoss()
        mse_fn = nn.MSELoss()
        step = 0
        model.train()
        train_iter = iter(train_loader)
        while step < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            c, t, r, m, emp, z_true, was_act, will_buy = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            opt.zero_grad()
            dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
            loss_r = bce_fn(r_logit[was_act == 0], will_buy[was_act == 0]) if (was_act == 0).sum() > 0 else torch.tensor(0.0, device=device)
            loss_c = bce_fn(c_logit[was_act == 1], (1.0 - will_buy[was_act == 1])) if (was_act == 1).sum() > 0 else torch.tensor(0.0, device=device)
            loss_cond = mse_fn(cond_z[will_buy == 1], z_true[will_buy == 1]) if (will_buy == 1).sum() > 0 else torch.tensor(0.0, device=device)
            total_loss = loss_r + loss_c + loss_cond
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        model.eval()
        r_logits, c_logits, cond_zs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                c, t, r, m, emp = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
                r_logits.append(r_logit.cpu().numpy())
                c_logits.append(c_logit.cpu().numpy())
                cond_zs.append(cond_z.cpu().numpy())
        return np.concatenate(r_logits), np.concatenate(c_logits), np.concatenate(cond_zs)

    print("\n[*] RUN 2: Training FINAL S1 Masked GRU Specialists...")
    s1_r_test, s1_c_test, s1_a_test = train_final_neural(GRUEncoder(d_in=12, d_model=128, n_layers=2), n_steps=2500, lr=5e-4)

    print("\n[*] RUN 2: Training FINAL S2 Dense GRU Specialists...")
    s2_r_test, s2_c_test, s2_a_test = train_final_neural(GRUEncoder(d_in=12, d_model=128, n_layers=2), n_steps=2500, lr=5e-4)

    print("\n[*] RUN 2: Training FINAL Event-Time Transformer Specialists...")
    ett_r_test, ett_c_test, ett_a_test = train_final_neural(EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=max_ev), n_steps=3500, lr=3e-4)

    # Assemble Final Predictions
    print("\n" + "=" * 80)
    print("FINAL SUBMISSION ASSEMBLY (CANONICAL FORMULA)")
    print("=" * 80)

    X_r_test = np.column_stack([cb_r_test, s1_r_test, s2_r_test, ett_r_test])
    X_c_test = np.column_stack([cb_c_test, s1_c_test, s2_c_test, ett_c_test])
    X_a_test = np.column_stack([cb_a_test, s1_a_test, s2_a_test, ett_a_test])

    p_r_test = expit(np.dot(X_r_test, w_react))
    p_c_test = expit(np.dot(X_c_test, w_churn))
    p_buy_test = np.where(was_act_test == 0, p_r_test, 1.0 - p_c_test)

    cond_z_test = np.maximum(0.0, ridge.predict(X_a_test))
    z_final = np.clip(np.power(p_buy_test, 1.1) * cond_z_test, 0.0, None)
    gmv_final = np.expm1(z_final)

    sub_file_name = "submission_specialized_hurdle_canonical_250k.csv"
    df_sub = pl.DataFrame({
        "user_id": all_users,
        "predict": gmv_final,
    })
    df_sub.write_csv(sub_file_name)
    df_sub.write_csv(out_dir / sub_file_name)
    print(f"\n[+] FINAL SUBMISSION SAVED TO {sub_file_name} ({len(df_sub):,} rows)")
    print(f"    Mean GMV:   {np.mean(gmv_final):.4f} RUB")
    print(f"    Median GMV: {np.median(gmv_final):.4f} RUB")
    print(f"    Min GMV:    {np.min(gmv_final):.4f} RUB")
    print(f"    Max GMV:    {np.max(gmv_final):.4f} RUB")

    with open(sub_file_name, "rb") as f:
        sub_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[+] SHA256: {sub_sha256}")
    print("[+] SUCCESS! All stages completed.")


if __name__ == "__main__":
    main()
