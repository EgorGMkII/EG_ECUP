#!/usr/bin/env python3
"""
scripts/specialized_hurdle_250k/run_temporal_regularized_meta.py

Full end-to-end pipeline for:
1. Training RUN A base specialists on cutoff <= 2025-12-15 (17 CatBoost anchors, 8 Neural anchors)
2. Generating 250k predictions on December anchor (2025-12-15) and January anchor (2026-01-14)
3. Grid search over lambda_cls x lambda_amount on December -> evaluated on January (Temporal Validation)
4. Paired bootstrap & One-Standard-Error rule lambda selection
5. Final fit on pooled December + January anchors (0.5 weight each)
6. Applying final regularized weights to test_specialists_raw_predictions_250k_v2.parquet
7. Generating submission_specialized_hurdle_joint_250k_temporal_regularized.csv
"""

import os
import gc
import json
import hashlib
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.special import expit
from catboost import CatBoostClassifier, CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. NEURAL ARCHITECTURES
# ==============================================================================

class GRUEncoder(nn.Module):
    def __init__(self, d_in: int = 12, d_model: int = 128, n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head_dir = nn.Linear(d_model, 1)
        self.head_cond = nn.Linear(d_model, 1)
        self.head_react = nn.Linear(d_model, 1)
        self.head_churn = nn.Linear(d_model, 1)

    def forward(self, content, time_feat, ranks, mask, empty):
        x = self.proj(content)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        h = torch.where(empty.unsqueeze(-1), torch.zeros_like(h), h)
        dir_z = torch.relu(self.head_dir(h)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(h)).squeeze(-1)
        r_logit = self.head_react(h).squeeze(-1)
        c_logit = self.head_churn(h).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


class EventTimeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        n_content_feats: int = 12,
        n_time_feats: int = 12,
        max_events: int = 180,
    ):
        super().__init__()
        self.cont_proj = nn.Linear(n_content_feats, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(n_time_feats, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.rank_emb = nn.Embedding(max_events + 1, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
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
            torch.from_numpy(np.asarray(self.ranks[idx], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.mask[idx], dtype=bool)),
            torch.tensor(bool(self.empty[idx]), dtype=torch.bool),
            torch.tensor(float(self.z_true[idx]), dtype=torch.float32),
            torch.tensor(float(self.was_active[idx]), dtype=torch.float32),
            torch.tensor(float(self.will_buy[idx]), dtype=torch.float32),
            torch.tensor(float(self.y_rub[idx]), dtype=torch.float32),
        )


# ==============================================================================
# 2. POLARS SEQUENCE EXTRACTION
# ==============================================================================

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

        take_n = min(n_ev, max_events)
        c_arr = np.column_stack([np.array(row[k][-take_n:], dtype=np.float32) for k in c_cols])
        t_arr = np.column_stack([np.array(row[k][-take_n:], dtype=np.float32) for k in t_cols])

        pad_len = max_events - take_n
        if pad_len > 0:
            out_c[idx, :pad_len] = 0.0
            out_t[idx, :pad_len] = 0.0
            out_r[idx, :pad_len] = 0
            out_m[idx, :pad_len] = True

        out_c[idx, pad_len:] = c_arr
        out_t[idx, pad_len:] = t_arr
        out_r[idx, pad_len:] = np.arange(1, take_n + 1, dtype=np.int16)
        out_m[idx, pad_len:] = False
        out_emp[idx] = False


# ==============================================================================
# 3. COMPUTE FUTURE GMV FOR ANY ANCHOR
# ==============================================================================

def compute_future_gmv_target(df_events: pl.DataFrame, user_ids: np.ndarray, anchor_str: str) -> np.ndarray:
    anchor_dt = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    target_start = anchor_dt + timedelta(days=1)
    target_end = anchor_dt + timedelta(days=30)

    df_future = df_events.filter(
        (pl.col("event_date") >= target_start) & (pl.col("event_date") <= target_end)
    ).group_by("user_id").agg(pl.col("gmv").sum().alias("future_gmv"))

    future_dict = {r["user_id"]: float(r["future_gmv"]) for r in df_future.iter_rows(named=True)}
    return np.array([future_dict.get(u, 0.0) for u in user_ids], dtype=np.float32)


# ==============================================================================
# 4. MAIN SCRIPT
# ==============================================================================

def main():
    print("=" * 80)
    print("TEMPORAL REGULARIZED META-OPTIMIZATION (250k USERS, DEC + JAN ANCHORS)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    base_dir = Path("artifacts/specialized_hurdle_250k/temporal_regularized_meta")
    cfg_dir = base_dir / "configs"
    pred_dir = base_dir / "predictions"
    opt_dir = base_dir / "optimization"
    diag_dir = base_dir / "diagnostics"
    rep_dir = base_dir / "reports"
    sub_dir = base_dir / "submissions"
    log_dir = base_dir / "logs"
    chk_dir = base_dir / "checkpoints"

    for d in [cfg_dir, pred_dir, opt_dir, diag_dir, rep_dir, sub_dir, log_dir, chk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load User Cohort (250,000 users)
    df_sample = pl.read_csv("sample_submit.csv")
    all_users = df_sample["user_id"].to_numpy()
    n_users = len(all_users)
    print(f"[+] Loaded 250k user cohort: {n_users:,} unique users.")

    # 2. Check existing December predictions
    dec_pred_path = Path("artifacts/specialized_hurdle_250k/joint_meta/meta_anchor_predictions_250k.parquet")
    if not dec_pred_path.exists():
        dec_pred_path = Path("meta_anchor_predictions_250k.parquet")

    # 3. Anchors definition
    anchor_dec = "2025-12-15"
    anchor_jan = "2026-01-14"

    run_a_cb_anchors = [
        "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
        "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
        "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
        "2025-10-27", "2025-11-10"
    ]
    run_a_neural_anchors = [
        "2025-03-31", "2025-04-28", "2025-05-26", "2025-06-23",
        "2025-07-21", "2025-08-18", "2025-09-15", "2025-10-27"
    ]

    events_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    print(f"[*] Loading raw events from {events_path}...")
    df_events = pl.read_parquet(events_path)
    print(f"[+] Loaded raw events: {len(df_events):,} rows.")

    snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")

    # ==========================================================================
    # STEP A: TRAIN RUN A SPECIALISTS (17 ANCHORS) & SAVE CHECKPOINTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP A: TRAINING RUN A SPECIALISTS (CUTOFF <= 2025-12-15)")
    print("=" * 80)

    cb_dfs = []
    for a in run_a_cb_anchors:
        p = snap_dir / f"snapshot_{a}.parquet"
        if p.exists():
            df_snap = pl.read_parquet(p)
            df_snap = df_snap.with_columns([
                pl.lit(a).alias("anchor_date"),
                (pl.col("lifetime_gmv") > 0).cast(pl.Int64).alias("was_active"),
                (pl.col("target") > 0).cast(pl.Int64).alias("will_buy"),
            ])
            cb_dfs.append(df_snap)
    df_cb_train = pl.concat(cb_dfs)
    print(f"[+] Pooled CatBoost training data: {len(df_cb_train):,} rows.")

    excluded = {
        "user_id", "target", "lifetime_gmv", "will_buy_30d", "anchor_date",
        "history_start", "history_end", "target_start", "target_end", "user_segment_id",
        "was_active", "will_buy", "future_gmv_30d", "available_history_days"
    }
    feat_cols = [c for c in df_cb_train.columns if c not in excluded and not c.startswith("future_") and not c.startswith("target")]
    print(f"[+] Clean feature count: {len(feat_cols)}")

    y_train_cb_gmv = df_cb_train["target"].to_numpy().astype(np.float32)
    was_act_tr = (df_cb_train["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_tr = (y_train_cb_gmv > 0).astype(int)
    X_cb_tr = df_cb_train.select(feat_cols).to_numpy().astype(np.float32)

    # 1. CB React
    mask_react = (was_act_tr == 0)
    print(f"\n[*] Training CB_REACT_META on {mask_react.sum():,} rows...")
    cb_react = CatBoostClassifier(
        iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=500,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    cb_react.fit(X_cb_tr[mask_react], will_buy_tr[mask_react])
    cb_react.save_model(str(chk_dir / "catboost_react.cbm"))

    # 2. CB Churn
    mask_churn = (was_act_tr == 1)
    print(f"\n[*] Training CB_CHURN_META on {mask_churn.sum():,} rows...")
    cb_churn = CatBoostClassifier(
        iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=500,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    cb_churn.fit(X_cb_tr[mask_churn], 1 - will_buy_tr[mask_churn])
    cb_churn.save_model(str(chk_dir / "catboost_churn.cbm"))

    # 3. CB Amount
    mask_amt = (y_train_cb_gmv > 0)
    print(f"\n[*] Training CB_AMOUNT_META on {mask_amt.sum():,} rows...")
    cb_amt = CatBoostRegressor(
        iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=42, verbose=500,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    cb_amt.fit(X_cb_tr[mask_amt], np.log1p(y_train_cb_gmv[mask_amt]))
    cb_amt.save_model(str(chk_dir / "catboost_amount.cbm"))

    del df_cb_train, X_cb_tr, was_act_tr, will_buy_tr, y_train_cb_gmv, cb_dfs
    gc.collect()

    # Neural sequence training
    print("\n[*] Extracting Neural Training Sequences (8 anchors)...")
    max_ev = 180
    n_neural_anchors = len(run_a_neural_anchors)
    total_samples = n_neural_anchors * 100000

    mmap_dir = Path("mmap_buffers_temporal")
    mmap_dir.mkdir(parents=True, exist_ok=True)

    train_c = np.memmap(mmap_dir / "train_c.dat", dtype=np.float16, mode="w+", shape=(total_samples, max_ev, 12))
    train_t = np.memmap(mmap_dir / "train_t.dat", dtype=np.float16, mode="w+", shape=(total_samples, max_ev, 12))
    train_r = np.memmap(mmap_dir / "train_r.dat", dtype=np.int16, mode="w+", shape=(total_samples, max_ev))
    train_m = np.memmap(mmap_dir / "train_m.dat", dtype=bool, mode="w+", shape=(total_samples, max_ev))
    train_m[:] = True
    train_emp = np.memmap(mmap_dir / "train_emp.dat", dtype=bool, mode="w+", shape=(total_samples,))
    train_emp[:] = False

    train_z = np.memmap(mmap_dir / "train_z.dat", dtype=np.float32, mode="w+", shape=(total_samples,))
    train_act = np.memmap(mmap_dir / "train_act.dat", dtype=np.float32, mode="w+", shape=(total_samples,))
    train_buy = np.memmap(mmap_dir / "train_buy.dat", dtype=np.float32, mode="w+", shape=(total_samples,))
    train_rub = np.memmap(mmap_dir / "train_rub.dat", dtype=np.float32, mode="w+", shape=(total_samples,))

    for a_idx, a_str in enumerate(run_a_neural_anchors):
        off = a_idx * 100000
        snap_a = pl.read_parquet(snap_dir / f"snapshot_{a_str}.parquet")
        u_list = snap_a["user_id"].to_numpy()
        y_r = snap_a["target"].to_numpy().astype(np.float32)
        past_g = snap_a["lifetime_gmv"].to_numpy().astype(np.float32)

        train_z[off : off + 100000] = np.log1p(np.maximum(0.0, y_r))
        train_act[off : off + 100000] = (past_g > 0).astype(np.float32)
        train_buy[off : off + 100000] = (y_r > 0).astype(np.float32)
        train_rub[off : off + 100000] = y_r

        extract_event_time_sequences(
            df_events, u_list, a_str, max_events=max_ev,
            out_c=train_c, out_t=train_t, out_r=train_r, out_m=train_m, out_emp=train_emp,
            offset=off
        )
        print(f"   -> Extracted sequences for {a_str} ({a_idx+1}/{n_neural_anchors})")

    train_ds = MemmapDataset(train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_buy, train_rub)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())

    def train_neural(model: nn.Module, n_steps: int = 4500, lr: float = 3e-4):
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
        return model

    print("\n[*] Training S1 Masked GRU Specialists...")
    s1_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s1_model = train_neural(s1_model, n_steps=4500, lr=5e-4)
    torch.save(s1_model.state_dict(), chk_dir / "s1_masked_gru.pt")

    print("\n[*] Training S2 Dense GRU Specialists...")
    s2_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s2_model = train_neural(s2_model, n_steps=4500, lr=3e-4)
    torch.save(s2_model.state_dict(), chk_dir / "s2_dense_gru.pt")

    print("\n[*] Training Event-Time Transformer Specialists...")
    ett_model = EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, dim_feedforward=256, dropout=0.10)
    ett_model = train_neural(ett_model, n_steps=4500, lr=3e-4)
    torch.save(ett_model.state_dict(), chk_dir / "event_time_transformer.pt")

    # ==========================================================================
    # STEP B: INFERENCE ON DECEMBER (2025-12-15) & JANUARY (2026-01-14) (250k USERS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP B: GENERATING 250k PREDICTIONS FOR DECEMBER & JANUARY META-ANCHORS")
    print("=" * 80)

    def infer_anchor(anchor_str: str) -> pl.DataFrame:
        print(f"\n[*] Running full 250k inference for anchor {anchor_str}...")
        import pandas as pd
        from src.snapshots import build_snapshot

        anchor_date_obj = pd.Timestamp(anchor_str).date()
        df_meta_snap = build_snapshot(
            data=df_events,
            user_ids=all_users.tolist(),
            anchor_date=anchor_date_obj,
            is_test=False,
        )

        missing_cols = [c for c in feat_cols if c not in df_meta_snap.columns]
        if missing_cols:
            df_meta_snap = df_meta_snap.with_columns([pl.lit(0.0).alias(c) for c in missing_cols])

        X_meta = df_meta_snap.select(feat_cols).to_numpy().astype(np.float32)
        lifetime_gmv = df_meta_snap["lifetime_gmv"].to_numpy().astype(np.float32)
        was_act = (lifetime_gmv > 0).astype(np.int64)

        future_gmv = df_meta_snap["target"].to_numpy().astype(np.float32)
        will_buy = (future_gmv > 0).astype(np.int64)
        transition = [f"{a}{b}" for a, b in zip(was_act, will_buy)]

        cb_r_logit = np.nan_to_num(cb_react.predict(X_meta, prediction_type="RawFormulaVal"), nan=0.0)
        cb_c_logit = np.nan_to_num(cb_churn.predict(X_meta, prediction_type="RawFormulaVal"), nan=0.0)
        cb_a_z = np.nan_to_num(cb_amt.predict(X_meta), nan=0.0)

        # Neural sequence extraction
        m_c = np.zeros((n_users, 180, 12), dtype=np.float32)
        m_t = np.zeros((n_users, 180, 12), dtype=np.float32)
        m_r = np.zeros((n_users, 180), dtype=np.int64)
        m_m = np.ones((n_users, 180), dtype=bool)
        m_emp = np.ones(n_users, dtype=bool)

        extract_event_time_sequences(
            df_events, all_users, anchor_str, max_events=180, tau_days=30.0,
            out_c=m_c, out_t=m_t, out_r=m_r, out_m=m_m, out_emp=m_emp, offset=0
        )

        s1_r_list, s1_c_list, s1_a_list = [], [], []
        s2_r_list, s2_c_list, s2_a_list = [], [], []
        ett_r_list, ett_c_list, ett_a_list = [], [], []

        bs = 1024
        s1_model.eval(); s2_model.eval(); ett_model.eval()

        with torch.no_grad():
            for i in range(0, n_users, bs):
                end_i = min(i + bs, n_users)
                c_b = torch.from_numpy(m_c[i:end_i]).to(device)
                t_b = torch.from_numpy(m_t[i:end_i]).to(device)
                r_b = torch.from_numpy(m_r[i:end_i]).to(device)
                m_b = torch.from_numpy(m_m[i:end_i]).to(device)
                emp_b = torch.from_numpy(m_emp[i:end_i]).to(device)

                c_b_masked = torch.where(m_b.unsqueeze(-1), torch.zeros_like(c_b), c_b)
                _, s1_a, s1_r, s1_c = s1_model(c_b_masked, t_b, r_b, m_b, emp_b)
                s1_r_list.append(s1_r.cpu().numpy())
                s1_c_list.append(s1_c.cpu().numpy())
                s1_a_list.append(s1_a.cpu().numpy())

                _, s2_a, s2_r, s2_c = s2_model(c_b, t_b, r_b, m_b, emp_b)
                s2_r_list.append(s2_r.cpu().numpy())
                s2_c_list.append(s2_c.cpu().numpy())
                s2_a_list.append(s2_a.cpu().numpy())

                _, ett_a, ett_r, ett_c = ett_model(c_b, t_b, r_b, m_b, emp_b)
                ett_r_list.append(ett_r.cpu().numpy())
                ett_c_list.append(ett_c.cpu().numpy())
                ett_a_list.append(ett_a.cpu().numpy())

        s1_r = np.nan_to_num(np.concatenate(s1_r_list), nan=0.0)
        s1_c = np.nan_to_num(np.concatenate(s1_c_list), nan=0.0)
        s1_a = np.nan_to_num(np.concatenate(s1_a_list), nan=0.0)

        s2_r = np.nan_to_num(np.concatenate(s2_r_list), nan=0.0)
        s2_c = np.nan_to_num(np.concatenate(s2_c_list), nan=0.0)
        s2_a = np.nan_to_num(np.concatenate(s2_a_list), nan=0.0)

        ett_r = np.nan_to_num(np.concatenate(ett_r_list), nan=0.0)
        ett_c = np.nan_to_num(np.concatenate(ett_c_list), nan=0.0)
        ett_a = np.nan_to_num(np.concatenate(ett_a_list), nan=0.0)

        df_out = pl.DataFrame({
            "user_id": all_users,
            "anchor": [anchor_str] * n_users,
            "was_active": was_act,
            "will_buy": will_buy,
            "future_gmv_30d": future_gmv,
            "transition": transition,
            "cb_react_logit": cb_r_logit,
            "s1_react_logit": s1_r,
            "s2_react_logit": s2_r,
            "ett_react_logit": ett_r,
            "cb_churn_logit": cb_c_logit,
            "s1_churn_logit": s1_c,
            "s2_churn_logit": s2_c,
            "ett_churn_logit": ett_c,
            "cb_amount_z": cb_a_z,
            "s1_amount_z": s1_a,
            "s2_amount_z": s2_a,
            "ett_amount_z": ett_a,
        })
        return df_out

    df_dec = infer_anchor(anchor_dec)
    df_jan = infer_anchor(anchor_jan)

    df_dec.write_parquet(pred_dir / f"meta_anchor_{anchor_dec}_predictions.parquet")
    df_jan.write_parquet(pred_dir / f"meta_anchor_{anchor_jan}_predictions.parquet")
    print(f"[+] Saved December predictions: {len(df_dec):,} rows to {pred_dir / f'meta_anchor_{anchor_dec}_predictions.parquet'}")
    print(f"[+] Saved January predictions:   {len(df_jan):,} rows to {pred_dir / f'meta_anchor_{anchor_jan}_predictions.parquet'}")

    # ==========================================================================
    # STEP C: TEMPORAL REGULARIZATION GRID SEARCH (DEC -> EVAL ON JAN)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP C: TEMPORAL REGULARIZATION GRID SEARCH")
    print("=" * 80)

    X_r_dec = df_dec.select(["cb_react_logit", "s1_react_logit", "s2_react_logit", "ett_react_logit"]).to_numpy()
    X_c_dec = df_dec.select(["cb_churn_logit", "s1_churn_logit", "s2_churn_logit", "ett_churn_logit"]).to_numpy()
    X_a_dec = df_dec.select(["cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z"]).to_numpy()
    act_dec = df_dec["was_active"].to_numpy()
    z_target_dec = np.log1p(np.maximum(df_dec["future_gmv_30d"].to_numpy(), 0.0))

    X_r_jan = df_jan.select(["cb_react_logit", "s1_react_logit", "s2_react_logit", "ett_react_logit"]).to_numpy()
    X_c_jan = df_jan.select(["cb_churn_logit", "s1_churn_logit", "s2_churn_logit", "ett_churn_logit"]).to_numpy()
    X_a_jan = df_jan.select(["cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z"]).to_numpy()
    act_jan = df_jan["was_active"].to_numpy()
    z_target_jan = np.log1p(np.maximum(df_jan["future_gmv_30d"].to_numpy(), 0.0))

    lambda_cls_grid = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    lambda_amt_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, None), (0.0, None), (0.0, None), (0.0, None),
        (None, None)
    ]
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[0:4]) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.sum(w[4:8]) - 1.0},
    ]

    starts = [
        # Uniform
        np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0]),
        # S1+ETT focused
        np.array([0.10, 0.40, 0.10, 0.40, 0.30, 0.20, 0.10, 0.40, 0.20, 0.10, 0.35, 0.35, 0.0]),
        # Perturbation
        np.array([0.20, 0.30, 0.10, 0.40, 0.25, 0.25, 0.10, 0.40, 0.15, 0.05, 0.40, 0.40, 0.05]),
    ]

    def compute_e2e_mse(X_r, X_c, X_a, was_act, z_target, w):
        p_r = expit(np.dot(X_r, w[0:4]))
        p_c = expit(np.dot(X_c, w[4:8]))
        p_buy = np.where(was_act == 0, p_r, 1.0 - p_c)
        cond_z = np.clip(np.dot(X_a, w[8:12]) + w[12], 0.0, None)
        z_pred = np.clip(np.power(p_buy, 1.1) * cond_z, 0.0, None)
        return float(np.mean((z_pred - z_target) ** 2))

    grid_results = []
    print(f"[*] Running {len(lambda_cls_grid) * len(lambda_amt_grid)} grid evaluations...")

    for l_cls in lambda_cls_grid:
        for l_amt in lambda_amt_grid:
            def objective(w):
                mse = compute_e2e_mse(X_r_dec, X_c_dec, X_a_dec, act_dec, z_target_dec, w)
                reg = l_cls * (np.sum(w[0:4]**2) + np.sum(w[4:8]**2)) + l_amt * (np.sum(w[8:12]**2) + w[12]**2)
                return mse + reg

            best_w = starts[0].copy()
            best_val = objective(best_w)

            for s in starts:
                try:
                    res = minimize(
                        objective, s, method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"maxiter": 300, "ftol": 1e-8}
                    )
                    if res is not None and hasattr(res, "fun") and not np.isnan(res.fun) and res.fun < best_val:
                        best_val = res.fun
                        best_w = res.x
                except Exception:
                    pass

            dec_mse = compute_e2e_mse(X_r_dec, X_c_dec, X_a_dec, act_dec, z_target_dec, best_w)
            dec_rmsle = np.sqrt(dec_mse)

            jan_mse = compute_e2e_mse(X_r_jan, X_c_jan, X_a_jan, act_jan, z_target_jan, best_w)
            jan_rmsle = np.sqrt(jan_mse)

            rec = {
                "lambda_cls": float(l_cls),
                "lambda_amount": float(l_amt),
                "december_mse": float(dec_mse),
                "december_rmsle": float(dec_rmsle),
                "january_mse": float(jan_mse),
                "january_rmsle": float(jan_rmsle),
                "gap_rmsle": float(abs(jan_rmsle - dec_rmsle)),
                "w_r_cb": float(best_w[0]),
                "w_r_s1": float(best_w[1]),
                "w_r_s2": float(best_w[2]),
                "w_r_ett": float(best_w[3]),
                "w_c_cb": float(best_w[4]),
                "w_c_s1": float(best_w[5]),
                "w_c_s2": float(best_w[6]),
                "w_c_ett": float(best_w[7]),
                "w_a_cb": float(best_w[8]),
                "w_a_s1": float(best_w[9]),
                "w_a_s2": float(best_w[10]),
                "w_a_ett": float(best_w[11]),
                "amount_intercept": float(best_w[12]),
                "weights_norm": float(np.linalg.norm(best_w)),
            }
            grid_results.append(rec)

    df_grid = pl.DataFrame(grid_results)
    df_grid.write_csv(opt_dir / "lambda_grid_results.csv")
    print(f"[+] Saved lambda grid search results ({len(df_grid)} rows) to {opt_dir / 'lambda_grid_results.csv'}")

    # ==========================================================================
    # STEP D: PAIRED BOOTSTRAP & ONE-STANDARD-ERROR SELECTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP D: PAIRED BOOTSTRAP & ONE-STANDARD-ERROR RULE")
    print("=" * 80)

    df_valid = df_grid.filter(pl.col("january_rmsle").is_not_nan() & pl.col("january_rmsle").is_not_null() & (pl.col("january_rmsle") > 0.0))
    if len(df_valid) == 0:
        df_valid = df_grid
    df_sorted = df_valid.sort("january_rmsle")
    top_candidates = df_sorted.head(10).to_dicts()

    best_cand = top_candidates[0]
    print(f"[+] Best January RMSLE candidate: lambda_cls={best_cand['lambda_cls']}, lambda_amt={best_cand['lambda_amount']} -> Jan RMSLE={best_cand['january_rmsle']:.6f}")

    n_boot = 300
    rng = np.random.RandomState(42)
    best_cand_w = np.array([
        best_cand["w_r_cb"], best_cand["w_r_s1"], best_cand["w_r_s2"], best_cand["w_r_ett"],
        best_cand["w_c_cb"], best_cand["w_c_s1"], best_cand["w_c_s2"], best_cand["w_c_ett"],
        best_cand["w_a_cb"], best_cand["w_a_s1"], best_cand["w_a_s2"], best_cand["w_a_ett"],
        best_cand["amount_intercept"]
    ])

    p_r_best = expit(np.dot(X_r_jan, best_cand_w[0:4]))
    p_c_best = expit(np.dot(X_c_jan, best_cand_w[4:8]))
    p_buy_best = np.where(act_jan == 0, p_r_best, 1.0 - p_c_best)
    cond_z_best = np.clip(np.dot(X_a_jan, best_cand_w[8:12]) + best_cand_w[12], 0.0, None)
    z_pred_best = np.clip(np.power(p_buy_best, 1.1) * cond_z_best, 0.0, None)
    sq_err_best = (z_pred_best - z_target_jan) ** 2

    boot_best_rmsles = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_users, size=n_users)
        boot_best_rmsles.append(np.sqrt(np.mean(sq_err_best[idx])))
    std_err_best = float(np.std(boot_best_rmsles))
    print(f"[+] Bootstrap Standard Error of best candidate: {std_err_best:.6f}")

    se_threshold = best_cand["january_rmsle"] + std_err_best
    print(f"[+] 1-SE Threshold: {se_threshold:.6f}")

    eligible = [c for c in top_candidates if not np.isnan(c["january_rmsle"]) and c["january_rmsle"] <= se_threshold]
    if not eligible:
        eligible = [best_cand]
    eligible_sorted = sorted(eligible, key=lambda x: (x["lambda_cls"] + x["lambda_amount"], -x["gap_rmsle"]), reverse=True)
    selected_cand = eligible_sorted[0]

    selected_l_cls = selected_cand["lambda_cls"]
    selected_l_amt = selected_cand["lambda_amount"]
    print(f"\n[+] SELECTED REGULARIZATION BY 1-SE RULE:")
    print(f"    lambda_cls:    {selected_l_cls}")
    print(f"    lambda_amount: {selected_l_amt}")
    print(f"    Jan RMSLE:     {selected_cand['january_rmsle']:.6f}")
    print(f"    Dec RMSLE:     {selected_cand['december_rmsle']:.6f}")

    selection_meta = {
        "best_raw_candidate": best_cand,
        "selected_candidate": selected_cand,
        "bootstrap_std_error": float(std_err_best),
        "one_se_threshold": float(se_threshold),
        "n_bootstrap_samples": n_boot,
    }
    with open(opt_dir / "regularization_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2)

    # ==========================================================================
    # STEP E: FINAL FIT ON POOLED DECEMBER + JANUARY ANCHORS (0.5 WEIGHT EACH)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP E: FINAL FIT ON POOLED DECEMBER + JANUARY ANCHORS")
    print("=" * 80)

    def combined_objective(w):
        mse_d = compute_e2e_mse(X_r_dec, X_c_dec, X_a_dec, act_dec, z_target_dec, w)
        mse_j = compute_e2e_mse(X_r_jan, X_c_jan, X_a_jan, act_jan, z_target_jan, w)
        comb_mse = 0.5 * mse_d + 0.5 * mse_j
        reg = selected_l_cls * (np.sum(w[0:4]**2) + np.sum(w[4:8]**2)) + selected_l_amt * (np.sum(w[8:12]**2) + w[12]**2)
        return comb_mse + reg

    best_final_w = starts[0].copy()
    best_final_val = combined_objective(best_final_w)

    for s in starts:
        try:
            res = minimize(
                combined_objective, s, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9}
            )
            if res is not None and hasattr(res, "fun") and not np.isnan(res.fun) and res.fun < best_final_val:
                best_final_val = res.fun
                best_final_w = res.x
        except Exception:
            pass

    final_w_react = best_final_w[0:4].tolist()
    final_w_churn = best_final_w[4:8].tolist()
    final_w_amount = best_final_w[8:12].tolist()
    final_b_amount = float(best_final_w[12])

    dec_final_mse = compute_e2e_mse(X_r_dec, X_c_dec, X_a_dec, act_dec, z_target_dec, best_final_w)
    jan_final_mse = compute_e2e_mse(X_r_jan, X_c_jan, X_a_jan, act_jan, z_target_jan, best_final_w)
    comb_final_rmsle = np.sqrt(0.5 * dec_final_mse + 0.5 * jan_final_mse)

    print("\n[+] FINAL REGULARIZED JOINT WEIGHTS:")
    print(f"    React Weights:  {[round(x, 4) for x in final_w_react]}")
    print(f"    Churn Weights:  {[round(x, 4) for x in final_w_churn]}")
    print(f"    Amount Coeffs:  {[round(x, 4) for x in final_w_amount]} + {final_b_amount:.4f}")
    print(f"    December RMSLE: {np.sqrt(dec_final_mse):.6f}")
    print(f"    January RMSLE:  {np.sqrt(jan_final_mse):.6f}")
    print(f"    Combined RMSLE: {comb_final_rmsle:.6f}")

    final_meta_pkg = {
        "experiment_name": "SPECIALIZED_HURDLE_JOINT_250K_TEMPORAL_REGULARIZED",
        "created_at": datetime.now().isoformat(),
        "lambda_cls": selected_l_cls,
        "lambda_amount": selected_l_amt,
        "react_stack_weights": final_w_react,
        "churn_stack_weights": final_w_churn,
        "amount_coefficients": final_w_amount,
        "amount_intercept": final_b_amount,
        "alpha_exponent": 1.1,
        "december_rmsle": float(np.sqrt(dec_final_mse)),
        "january_rmsle": float(np.sqrt(jan_final_mse)),
        "combined_rmsle": float(comb_final_rmsle),
        "model_order": ["CatBoost", "S1_Masked_GRU", "S2_Dense_GRU", "Event_Time_Transformer"],
    }
    with open(opt_dir / "joint_meta_weights_250k_temporal_regularized.json", "w") as f:
        json.dump(final_meta_pkg, f, indent=2)
    with open("joint_meta_weights_250k_temporal_regularized.json", "w") as f:
        json.dump(final_meta_pkg, f, indent=2)

    with open(opt_dir / "model_order.json", "w") as f:
        json.dump({"model_order": ["CatBoost", "S1_Masked_GRU", "S2_Dense_GRU", "Event_Time_Transformer"]}, f, indent=2)

    # Copy files to root for DataSphere output collection
    df_grid.write_csv("lambda_grid_results.csv")
    with open("regularization_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2)
    df_dec.write_parquet("meta_anchor_2025-12-15_predictions.parquet")
    df_jan.write_parquet("meta_anchor_2026-01-14_predictions.parquet")

    # ==========================================================================
    # STEP F: APPLY TO TEST PREDICTIONS (test_specialists_raw_predictions_250k_v2.parquet)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP F: APPLYING REGULARIZED META-WEIGHTS TO FINAL 250k TEST PREDICTIONS")
    print("=" * 80)

    test_raw_path = Path("test_specialists_raw_predictions_250k_v2.parquet")
    if not test_raw_path.exists():
        test_raw_path = Path("artifacts/specialized_hurdle_250k/final_250k/predictions/test_specialists_raw_predictions_250k_v2.parquet")

    df_test_raw = pl.read_parquet(test_raw_path)
    print(f"[+] Loaded test specialist predictions: {len(df_test_raw):,} rows.")

    X_r_test = df_test_raw.select(["cb_react_logit", "s1_react_logit", "s2_react_logit", "ett_react_logit"]).to_numpy()
    X_c_test = df_test_raw.select(["cb_churn_logit", "s1_churn_logit", "s2_churn_logit", "ett_churn_logit"]).to_numpy()
    X_a_test = df_test_raw.select(["cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z"]).to_numpy()
    was_act_test = df_test_raw["was_active"].to_numpy()

    p_r_test = expit(np.dot(X_r_test, np.array(final_w_react)))
    p_c_test = expit(np.dot(X_c_test, np.array(final_w_churn)))
    p_buy_test = np.where(was_act_test == 0, p_r_test, 1.0 - p_c_test)

    cond_z_test = np.clip(np.dot(X_a_test, np.array(final_w_amount)) + final_b_amount, 0.0, None)
    z_final = np.clip(np.power(p_buy_test, 1.1) * cond_z_test, 0.0, None)
    gmv_final = np.expm1(z_final)

    df_sub = pl.DataFrame({
        "user_id": df_test_raw["user_id"],
        "predict": gmv_final,
    })

    sub_file_name = "submission_specialized_hurdle_joint_250k_temporal_regularized.csv"
    df_sub.write_csv(sub_file_name)
    df_sub.write_csv(sub_dir / sub_file_name)
    print(f"\n[+] FINAL REGULARIZED SUBMISSION SAVED TO {sub_file_name} and {sub_dir / sub_file_name}")

    # Compute SHA256
    with open(sub_file_name, "rb") as f:
        sub_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[+] SHA256 of {sub_file_name}: {sub_sha256}")

    # Diagnostics table
    df_diag_test = pl.DataFrame({
        "user_id": df_test_raw["user_id"],
        "was_active": was_act_test,
        "p_react": p_r_test,
        "p_churn": p_c_test,
        "p_buy": p_buy_test,
        "conditional_z": cond_z_test,
        "z_prediction": z_final,
        "predict": gmv_final,
    })
    df_diag_test.write_parquet(diag_dir / "test_predictions_diagnostics.parquet")

    # Update run_status.json
    status_pkg = {
        "stage": "READY_FOR_PUBLIC_AB_TEST",
        "timestamp": datetime.now().isoformat(),
        "submission_file": sub_file_name,
        "sha256": sub_sha256,
        "december_rmsle": float(np.sqrt(dec_final_mse)),
        "january_rmsle": float(np.sqrt(jan_final_mse)),
        "combined_rmsle": float(comb_final_rmsle),
        "lambda_cls": selected_l_cls,
        "lambda_amount": selected_l_amt,
        "stats": {
            "mean_gmv": float(np.mean(gmv_final)),
            "median_gmv": float(np.median(gmv_final)),
            "min_gmv": float(np.min(gmv_final)),
            "max_gmv": float(np.max(gmv_final)),
        }
    }
    with open(log_dir / "run_status.json", "w") as f:
        json.dump(status_pkg, f, indent=2)

    print("\n[+] SUCCESS! Stage status: READY_FOR_PUBLIC_AB_TEST")


if __name__ == "__main__":
    main()
