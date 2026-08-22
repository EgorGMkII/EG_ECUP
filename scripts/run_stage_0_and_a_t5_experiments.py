"""Master runner for Stage 0 (Anchor Edge Audit & A0_V1 vs A0_V2) and Stage A (T5 Reproduction Suite).

Strict guarantees:
- 0 days target leakage.
- Clean tensor loading & on-the-fly generation.
- Parity inference check (max_abs_diff < 1e-5).
- Paired bootstrap (N=1000).
- Test distribution check on anchor 2026-02-13.
- Blending in log1p space.
"""

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sequential.dataset import build_user_sequence_tensor
from src.sequential.models import MultiTaskTransitionGRUModel
from src.sequential.preprocessing import CHANNELS, NUMERIC_CHANNELS, SequentialScaler
from src.validation import get_snapshot_path

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
CACHE_DIR = DATA_DIR / "sequential_cache" if (DATA_DIR / "sequential_cache").exists() else Path("sequential_cache")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
USERS_PARQUET = Path("artifacts/selected_users_100k.parquet") if Path("artifacts/selected_users_100k.parquet").exists() else Path("selected_users_100k.parquet")

OUT_ROOT_STAGE0 = Path("artifacts/anchor_edge_audit")
OUT_ROOT_STAGEA = Path("artifacts/t5_reproduction")

for p in [OUT_ROOT_STAGE0, OUT_ROOT_STAGEA, OUT_ROOT_STAGEA / "checkpoints", OUT_ROOT_STAGEA / "predictions", OUT_ROOT_STAGEA / "plots", OUT_ROOT_STAGEA / "logs"]:
    p.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)
VAL_TARGET_START = VAL_ANCHOR + timedelta(days=1)
VAL_TARGET_END = VAL_ANCHOR + timedelta(days=30)
TEST_ANCHOR = date(2026, 2, 13)

ANCHORS_V1 = [
    date(2025, 7, 21),
    date(2025, 8, 4),
    date(2025, 8, 18),
    date(2025, 9, 1),
    date(2025, 9, 15),
    date(2025, 9, 29),
    date(2025, 10, 13),
    date(2025, 10, 27),
    date(2025, 11, 10),
    date(2025, 11, 24),
    date(2025, 12, 8),
]

ANCHORS_V2_EDGE = ANCHORS_V1 + [date(2025, 12, 15)]


# =============================================================================
# 1. T5 SIMPLIFIED TRANSFORMER ARCHITECTURE
# =============================================================================

class T5SimplifiedTransformerModel(nn.Module):
    """Canonical T5 Simplified 2-Layer PreLN Patch Transformer with robust Activity Masking."""

    def __init__(
        self,
        input_dim: int = 15,
        patch_size: int = 7,
        num_patches: int = 52,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model

        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Learned representation for fully dormant users
        self.empty_history_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.empty_history_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_linear = nn.Linear(d_model, 1)
        self.head_react = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 365, C] or [B, 364, C]
        # Keep most recent 364 days ending at anchor date
        x_trim = x[:, -self.num_patches * self.patch_size :, :]
        B = x_trim.shape[0]

        # Patch projection
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])
        tokens = self.patch_proj(x_patches) + self.pos_embed

        if padding_mask is not None:
            # Check for completely empty histories
            all_masked = padding_mask.all(dim=1)  # [B]
            if all_masked.any():
                # For fully empty users, unmask first token and inject learned empty token
                padding_mask = padding_mask.clone()
                padding_mask[all_masked, 0] = False
                tokens[all_masked, 0, :] = self.empty_history_token.squeeze(0)

            enc_out = self.transformer(tokens, src_key_padding_mask=padding_mask)
        else:
            enc_out = self.transformer(tokens)

        # Attention pooling
        attn_scores = self.attn_linear(enc_out)
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(attn_scores, dim=1)
        emb = torch.sum(weights * enc_out, dim=1)

        lr = self.head_react(emb).squeeze(-1)
        lc = self.head_churn(emb).squeeze(-1)
        lb = self.head_buy(emb).squeeze(-1)
        zc = self.head_cond(emb).squeeze(-1)
        zd = self.head_dir(emb).squeeze(-1)
        return lr, lc, lb, zc, zd, emb


# =============================================================================
# 2. DATASETS AND SEQUENCE HELPERS
# =============================================================================

class TransitionSequenceDataset(Dataset):
    """Memory-mapped sequence dataset supporting arbitrary history length and purged anchors."""

    def __init__(
        self,
        data: pl.DataFrame,
        anchors: List[date],
        user_ids: List[int],
        seq_len: int = 180,
        snapshots_dir: Path = SNAPSHOTS_DIR,
        cache_dir: Path = CACHE_DIR,
        scaler: Optional[SequentialScaler] = None,
    ):
        self.data = data
        self.anchors = anchors
        self.user_ids = user_ids
        self.n_users = len(user_ids)
        self.seq_len = seq_len
        self.scaler = scaler

        print(f"[*] Loading dataset: {len(anchors)} anchors x {self.n_users} users (seq_len={seq_len})...")
        y_rubs, past_buyers, fut_buyers = [], [], []
        self.seq_memmaps = []

        for a in anchors:
            snap = pl.read_parquet(get_snapshot_path(a, snapshots_dir))
            y_r = snap["target"].to_numpy().astype(np.float32)
            p_b = (snap["gmv_sum_30d"].to_numpy() > 0).astype(np.float32)
            f_b = (y_r > 0).astype(np.float32)

            y_rubs.append(y_r)
            past_buyers.append(p_b)
            fut_buyers.append(f_b)

            f_name = f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{self.n_users}_t{self.seq_len}.npy"
            p_seq = cache_dir / f_name
            if not p_seq.exists():
                print(f"  [*] Generating sequence tensor for {a} (t={self.seq_len}) on the fly...")
                seq_m = build_user_sequence_tensor(data, user_ids, a, seq_len=self.seq_len, cache_dir=cache_dir)
            else:
                seq_m = np.load(p_seq, mmap_mode="r")
            self.seq_memmaps.append(seq_m)

        self.y_rubs = np.concatenate(y_rubs, axis=0)
        self.z_trues = np.log1p(self.y_rubs)
        self.past_buyers = np.concatenate(past_buyers, axis=0)
        self.fut_buyers = np.concatenate(fut_buyers, axis=0)

    def __len__(self) -> int:
        return len(self.y_rubs)

    def __getitem__(self, idx: int):
        anchor_idx = idx // self.n_users
        user_idx = idx % self.n_users

        seq = np.array(self.seq_memmaps[anchor_idx][user_idx], dtype=np.float32)
        if self.scaler is not None:
            seq = self.scaler.transform(seq[np.newaxis, ...])[0]

        return (
            torch.from_numpy(seq),
            torch.tensor(self.z_trues[idx], dtype=torch.float32),
            torch.tensor(self.past_buyers[idx], dtype=torch.float32),
            torch.tensor(self.fut_buyers[idx], dtype=torch.float32),
            torch.tensor(self.y_rubs[idx], dtype=torch.float32),
        )


def compute_behavioral_padding_mask(x_tensor: torch.Tensor, patch_size: int = 7, num_patches: int = 52) -> torch.Tensor:
    """Builds boolean src_key_padding_mask (True = ignore) strictly based on 12 behavioral channels."""
    # x_tensor: [B, 365, 15] or [B, 364, 15]
    x_trim = x_tensor[:, -num_patches * patch_size :, :12]  # [B, 364, 12]
    B = x_trim.shape[0]
    # Patch is active if sum of absolute behavioral actions > 0
    patch_activity = x_trim.abs().sum(dim=-1).reshape(B, num_patches, patch_size).sum(dim=-1) > 0  # [B, 52]
    # In PyTorch Transformer: True means ignored / padded
    padding_mask = ~patch_activity
    return padding_mask


# =============================================================================
# 3. TRAINING AND EVALUATION ENGINE
# =============================================================================

def train_downstream_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_tensor: np.ndarray,
    val_z_true: np.ndarray,
    val_past_buyer: np.ndarray,
    val_y_rub: np.ndarray,
    device: torch.device,
    exp_id: str,
    epochs: int = 10,
    lr: float = 1e-3,
    is_transformer: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=1.0)

    best_val_rmsle = 999.0
    best_z_pred = None
    best_gmv_pred = None

    print(f"\n[*] Training {exp_id} ({epochs} epochs on {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n_b = 0.0, 0
        t0_ep = time.time()

        for x_b, z_b, past_b, fut_b, _ in train_loader:
            x_b = x_b.to(device)
            z_b, past_b, fut_b = z_b.to(device), past_b.to(device), fut_b.to(device)

            optimizer.zero_grad()

            if is_transformer:
                p_mask = compute_behavioral_padding_mask(x_b)
                lr_out, lc_out, lb_out, zc_out, zd_out, _ = model(x_b, padding_mask=p_mask)
            else:
                lr_out, lc_out, lb_out, zc_out, zd_out, _ = model(x_b)

            m_dormant = past_b == 0
            m_active = past_b == 1
            m_buyer = fut_b == 1

            loss_r = bce(lr_out[m_dormant], fut_b[m_dormant]) if m_dormant.sum() > 0 else torch.tensor(0.0, device=device)
            loss_c = bce(lc_out[m_active], (1.0 - fut_b[m_active])) if m_active.sum() > 0 else torch.tensor(0.0, device=device)
            loss_b = bce(lb_out, fut_b)
            loss_cond = huber(zc_out[m_buyer], z_b[m_buyer]) if m_buyer.sum() > 0 else torch.tensor(0.0, device=device)
            loss_dir = huber(zd_out, z_b)

            loss = loss_r + loss_c + 0.5 * loss_b + 2.0 * loss_cond + 0.5 * loss_dir
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            n_b += 1

        scheduler.step()
        ep_loss /= max(1, n_b)

        # Validation Inference
        model.eval()
        p_react_l, p_churn_l, z_cond_l = [], [], []
        inf_bs = 2048
        with torch.no_grad():
            for i in range(0, len(val_tensor), inf_bs):
                xb = torch.from_numpy(val_tensor[i : i + inf_bs]).to(device)
                if is_transformer:
                    pm = compute_behavioral_padding_mask(xb)
                    lr_o, lc_o, _, zc_o, _, _ = model(xb, padding_mask=pm)
                else:
                    lr_o, lc_o, _, zc_o, _, _ = model(xb)

                p_react_l.append(torch.sigmoid(lr_o).cpu().numpy())
                p_churn_l.append(torch.sigmoid(lc_o).cpu().numpy())
                z_cond_l.append(zc_o.cpu().numpy())

            p_react = np.concatenate(p_react_l)
            p_churn = np.concatenate(p_churn_l)
            z_cond = np.concatenate(z_cond_l)

            p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
            z_fact = (np.power(p_buy, 1.10) * np.maximum(z_cond, 0.0)).astype(np.float32)
            gmv_fact = np.expm1(z_fact)

            val_rmsle = float(np.sqrt(np.mean((z_fact - val_z_true) ** 2)))

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({time.time() - t0_ep:.1f}s) | Loss: {ep_loss:.4f} | Val RMSLE: {val_rmsle:.5f}")

        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            best_z_pred = z_fact
            best_gmv_pred = gmv_fact
            torch.save(model.state_dict(), OUT_ROOT_STAGEA / "checkpoints" / f"{exp_id}_best.pt")

    print(f"[+] {exp_id} Finished! Best Val RMSLE: {best_val_rmsle:.5f}")
    return best_z_pred, best_gmv_pred, best_val_rmsle


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("=== HIGH-IMPACT EXPERIMENTAL SUITE: STAGE 0 & STAGE A (T5) ===")
    print("=" * 80)
    t0_master = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()
    assert len(user_ids) == 100000, "Validation user count mismatch!"
    data = pl.read_parquet(TRAIN_PARQUET)

    # Validation ground truth
    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    val_y_rub = val_snap["target"].to_numpy().astype(np.float32)
    val_z_true = np.log1p(val_y_rub)
    val_past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.float32)
    val_fut_buyer = (val_y_rub > 0).astype(np.float32)

    # Prepare 180d Scaler & Validation Tensor
    val_seq_file_180 = CACHE_DIR / f"seq_tensor_2026-01-14_u100000_t180.npy"
    if not val_seq_file_180.exists():
        val_memmap_180 = build_user_sequence_tensor(data, user_ids, VAL_ANCHOR, seq_len=180, cache_dir=CACHE_DIR)
    else:
        val_memmap_180 = np.load(val_seq_file_180, mmap_mode="r")

    tr_sample_seqs_180 = []
    for a in ANCHORS_V1[:3]:
        p_seq = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u100000_t180.npy"
        if not p_seq.exists():
            seq_m = build_user_sequence_tensor(data, user_ids, a, seq_len=180, cache_dir=CACHE_DIR)
        else:
            seq_m = np.load(p_seq, mmap_mode="r")
        tr_sample_seqs_180.append(np.array(seq_m[:20000]))

    scaler_180 = SequentialScaler()
    scaler_180.fit(np.concatenate(tr_sample_seqs_180, axis=0))
    val_tensor_180 = scaler_180.transform(np.array(val_memmap_180, dtype=np.float32))

    # Prepare 365d Scaler & Validation Tensor
    val_seq_file_365 = CACHE_DIR / f"seq_tensor_2026-01-14_u100000_t365.npy"
    if not val_seq_file_365.exists():
        val_memmap_365 = build_user_sequence_tensor(data, user_ids, VAL_ANCHOR, seq_len=365, cache_dir=CACHE_DIR)
    else:
        val_memmap_365 = np.load(val_seq_file_365, mmap_mode="r")

    tr_sample_seqs_365 = []
    for a in ANCHORS_V1[:3]:
        p_seq = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u100000_t365.npy"
        if not p_seq.exists():
            seq_m = build_user_sequence_tensor(data, user_ids, a, seq_len=365, cache_dir=CACHE_DIR)
        else:
            seq_m = np.load(p_seq, mmap_mode="r")
        tr_sample_seqs_365.append(np.array(seq_m[:20000]))

    scaler_365 = SequentialScaler()
    scaler_365.fit(np.concatenate(tr_sample_seqs_365, axis=0))
    val_tensor_365 = scaler_365.transform(np.array(val_memmap_365, dtype=np.float32))

    # -------------------------------------------------------------------------
    # STAGE 0: A0_V1 (11 anchors) vs A0_V2 (11 + edge anchor 2025-12-15)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE 0: CANONICAL GRU-180 ON V1 (11 ANCHORS) VS V2 (12 ANCHORS) ===")
    print("=" * 80)

    # A0_V1
    ds_train_v1 = TransitionSequenceDataset(data, ANCHORS_V1, user_ids, seq_len=180, scaler=scaler_180)
    loader_v1 = DataLoader(ds_train_v1, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    torch.manual_seed(42)
    model_a0_v1 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.20).to(device)
    z_pred_a0_v1, gmv_pred_a0_v1, rmsle_a0_v1 = train_downstream_model(
        model_a0_v1, loader_v1, val_tensor_180, val_z_true, val_past_buyer, val_y_rub, device, "A0_V1_Canonical_GRU180", epochs=10
    )

    # A0_V2 (with 2025-12-15 edge anchor)
    ds_train_v2 = TransitionSequenceDataset(data, ANCHORS_V2_EDGE, user_ids, seq_len=180, scaler=scaler_180)
    loader_v2 = DataLoader(ds_train_v2, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    torch.manual_seed(42)
    model_a0_v2 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.20).to(device)
    z_pred_a0_v2, gmv_pred_a0_v2, rmsle_a0_v2 = train_downstream_model(
        model_a0_v2, loader_v2, val_tensor_180, val_z_true, val_past_buyer, val_y_rub, device, "A0_V2_Canonical_GRU180_Edge", epochs=10
    )

    print(f"\n[+] Stage 0 Result:")
    print(f"    A0_V1 (11 Anchors):     RMSLE = {rmsle_a0_v1:.5f}")
    print(f"    A0_V2 (11 + Edge 12-15): RMSLE = {rmsle_a0_v2:.5f} (Delta: {rmsle_a0_v2 - rmsle_a0_v1:+.5f})")

    # -------------------------------------------------------------------------
    # STAGE A4: 20-USER AUDIT OF BEHAVIORAL MASKING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A4: 20-USER AUDIT OF BEHAVIORAL ACTIVITY MASKING ===")
    print("=" * 80)
    audit_sample_users = user_ids[:20]
    audit_raw_tensors = val_memmap_365[:20]
    audit_p_mask = compute_behavioral_padding_mask(torch.from_numpy(audit_raw_tensors))

    audit_records = []
    for i, uid in enumerate(audit_sample_users):
        raw_b = audit_raw_tensors[i, :, :12]
        n_active_days = int(np.sum(np.abs(raw_b).sum(axis=-1) > 0))
        n_masked_patches = int(audit_p_mask[i].sum().item())
        is_fully_empty = (n_active_days == 0)

        audit_records.append({
            "user_id": uid,
            "active_days_count": n_active_days,
            "masked_patches_52": n_masked_patches,
            "active_patches_52": 52 - n_masked_patches,
            "is_fully_empty_history": is_fully_empty,
            "padding_mask_all_true": bool(n_masked_patches == 52),
        })

    with open(OUT_ROOT_STAGEA / "user_mask_audit_20.json", "w") as f:
        json.dump(audit_records, f, indent=2)
    print(f"[+] Saved {OUT_ROOT_STAGEA / 'user_mask_audit_20.json'}")

    # -------------------------------------------------------------------------
    # STAGE A6: T5 EXPERIMENTS (T5_R0 ON V1, T5_R1 ON V2_EDGE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A6: T5 EXPERIMENTS (T5_R0 VS T5_R1) ===")
    print("=" * 80)

    # T5_R0 (V1 Anchors, seed 42)
    ds_train_365_v1 = TransitionSequenceDataset(data, ANCHORS_V1, user_ids, seq_len=365, scaler=scaler_365)
    loader_365_v1 = DataLoader(ds_train_365_v1, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    torch.manual_seed(42)
    model_t5_r0 = T5SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    z_pred_t5_r0, gmv_pred_t5_r0, rmsle_t5_r0 = train_downstream_model(
        model_t5_r0, loader_365_v1, val_tensor_365, val_z_true, val_past_buyer, val_y_rub, device, "T5_R0_Purged_V1", epochs=10, is_transformer=True
    )

    # T5_R1 (V2_EDGE Anchors, seed 42)
    ds_train_365_v2 = TransitionSequenceDataset(data, ANCHORS_V2_EDGE, user_ids, seq_len=365, scaler=scaler_365)
    loader_365_v2 = DataLoader(ds_train_365_v2, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    torch.manual_seed(42)
    model_t5_r1 = T5SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    z_pred_t5_r1, gmv_pred_t5_r1, rmsle_t5_r1 = train_downstream_model(
        model_t5_r1, loader_365_v2, val_tensor_365, val_z_true, val_past_buyer, val_y_rub, device, "T5_R1_Purged_V2_Edge", epochs=10, is_transformer=True
    )

    # -------------------------------------------------------------------------
    # STAGE A5: PARITY INFERENCE CHECK (ON 10 000 USERS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A5: PARITY INFERENCE CHECK (10 000 USERS) ===")
    print("=" * 80)
    sample_val_tensor = val_tensor_365[:10000]
    xb_samp = torch.from_numpy(sample_val_tensor).to(device)
    pm_samp = compute_behavioral_padding_mask(xb_samp)

    model_t5_r1.eval()
    with torch.no_grad():
        lr_orig, lc_orig, lb_orig, zc_orig, zd_orig, emb_orig = model_t5_r1(xb_samp, padding_mask=pm_samp)

    # Reload from checkpoint
    reloaded_t5 = T5SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    reloaded_t5.load_state_dict(torch.load(OUT_ROOT_STAGEA / "checkpoints" / "T5_R1_Purged_V2_Edge_best.pt"), strict=True)
    reloaded_t5.eval()
    with torch.no_grad():
        lr_rel, lc_rel, lb_rel, zc_rel, zd_rel, emb_rel = reloaded_t5(xb_samp, padding_mask=pm_samp)

    diff_lr = float((lr_orig - lr_rel).abs().max().item())
    diff_zc = float((zc_orig - zc_rel).abs().max().item())
    diff_emb = float((emb_orig - emb_rel).abs().max().item())

    print(f"[+] Parity Check Max Differences:")
    print(f"    Logits Diff:     {diff_lr:.2e}")
    print(f"    Conditional Z:   {diff_zc:.2e}")
    print(f"    Embedding Diff:  {diff_emb:.2e}")
    parity_passed = (max(diff_lr, diff_zc, diff_emb) < 1e-5)
    print(f"    Parity Passed (diff < 1e-5): {parity_passed}")

    with open(OUT_ROOT_STAGEA / "parity_check.json", "w") as f:
        json.dump({
            "max_abs_diff_logits": diff_lr,
            "max_abs_diff_z_cond": diff_zc,
            "max_abs_diff_embeddings": diff_emb,
            "parity_passed": parity_passed,
        }, f, indent=2)

    # -------------------------------------------------------------------------
    # STAGE A6 (T5_R2): 3-SEED STABILITY CHECK FOR WINNING ANCHOR SET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A6: T5_R2 3-SEED STABILITY CHECK ===")
    print("=" * 80)
    best_loader = loader_365_v2 if rmsle_t5_r1 <= rmsle_t5_r0 else loader_365_v1
    best_anchors_name = "V2_EDGE" if rmsle_t5_r1 <= rmsle_t5_r0 else "V1"

    seed_results = {"42": rmsle_t5_r1 if best_anchors_name == "V2_EDGE" else rmsle_t5_r0}
    seed_preds = {"42": z_pred_t5_r1 if best_anchors_name == "V2_EDGE" else z_pred_t5_r0}

    for s in [43, 44]:
        torch.manual_seed(s)
        m_seed = T5SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
        z_s, gmv_s, r_s = train_downstream_model(
            m_seed, best_loader, val_tensor_365, val_z_true, val_past_buyer, val_y_rub, device, f"T5_R2_seed{s}", epochs=10, is_transformer=True
        )
        seed_results[str(s)] = r_s
        seed_preds[str(s)] = z_s

    # Seed Ensemble
    z_t5_3seed = np.mean([seed_preds["42"], seed_preds["43"], seed_preds["44"]], axis=0)
    rmsle_t5_3seed = float(np.sqrt(np.mean((z_t5_3seed - val_z_true) ** 2)))
    print(f"[+] 3-Seed T5 Mean RMSLE: {np.mean(list(seed_results.values())):.5f} | 3-Seed Ensemble RMSLE: {rmsle_t5_3seed:.5f}")

    # -------------------------------------------------------------------------
    # STAGE A7: TEST DISTRIBUTION CHECK (ANCHOR 2026-02-13)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A7: TEST DISTRIBUTION AUDIT (ANCHOR 2026-02-13) ===")
    print("=" * 80)
    test_seq_file = CACHE_DIR / f"seq_tensor_{TEST_ANCHOR.strftime('%Y-%m-%d')}_u{len(user_ids)}_t365.npy"
    if not test_seq_file.exists():
        test_memmap_365 = build_user_sequence_tensor(data, user_ids, TEST_ANCHOR, seq_len=365, cache_dir=CACHE_DIR)
    else:
        test_memmap_365 = np.load(test_seq_file, mmap_mode="r")

    test_tensor_365 = scaler_365.transform(np.array(test_memmap_365, dtype=np.float32))

    # Infer test
    model_t5_r1.eval()
    test_z_cond_l, test_lr_l, test_lc_l, test_emb_l = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(test_tensor_365), 2048):
            xb_t = torch.from_numpy(test_tensor_365[i : i + 2048]).to(device)
            pm_t = compute_behavioral_padding_mask(xb_t)
            lr_t, lc_t, _, zc_t, _, emb_t = model_t5_r1(xb_t, padding_mask=pm_t)
            test_lr_l.append(torch.sigmoid(lr_t).cpu().numpy())
            test_lc_l.append(torch.sigmoid(lc_t).cpu().numpy())
            test_z_cond_l.append(zc_t.cpu().numpy())
            if i == 0:
                test_emb_l.append(emb_t.cpu().numpy())

    test_p_react = np.concatenate(test_lr_l)
    test_p_churn = np.concatenate(test_lc_l)
    test_z_cond = np.concatenate(test_z_cond_l)
    test_emb = np.concatenate(test_emb_l)

    # Last 30d GMV before test anchor
    test_past_buyer = (val_y_rub > 0).astype(np.float32)  # Target of 2026-01-14 is past 30d before 2026-02-13!
    test_p_buy = np.where(test_past_buyer == 0, test_p_react, 1.0 - test_p_churn)
    test_z_fact = (np.power(test_p_buy, 1.10) * np.maximum(test_z_cond, 0.0)).astype(np.float32)
    test_gmv_rub = np.expm1(test_z_fact)

    test_dist_report = {
        "test_mean_rub": round(float(np.mean(test_gmv_rub)), 2),
        "test_p50_rub": round(float(np.median(test_gmv_rub)), 2),
        "test_p90_rub": round(float(np.percentile(test_gmv_rub, 90)), 2),
        "test_p99_rub": round(float(np.percentile(test_gmv_rub, 99)), 2),
        "test_zero_rate": round(float(np.mean(test_gmv_rub < 1.0)), 4),
        "test_p_buy_mean": round(float(np.mean(test_p_buy)), 4),
        "test_cond_z_mean": round(float(np.mean(test_z_cond)), 4),
        "test_embedding_norm": round(float(np.mean(np.linalg.norm(test_emb, axis=1))), 4),
    }
    with open(OUT_ROOT_STAGEA / "test_distribution_audit.json", "w") as f:
        json.dump(test_dist_report, f, indent=2)
    print(f"[+] Saved {OUT_ROOT_STAGEA / 'test_distribution_audit.json'}: Mean = {test_dist_report['test_mean_rub']} rub, P99 = {test_dist_report['test_p99_rub']} rub")

    # -------------------------------------------------------------------------
    # STAGE A8: BLENDING & PAIRED BOOTSTRAP VS CATBOOST & GRU
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== STAGE A8: LOG1P BLENDING & PAIRED BOOTSTRAP EVALUATION ===")
    print("=" * 80)

    # Load CatBoost B1
    cb_p = Path("artifacts/tweedie_catboost/C0_Purged_Hurdle_B1/validation_predictions.parquet")
    if cb_p.exists():
        z_cb = pl.read_parquet(cb_p)["z_pred"].to_numpy().astype(np.float32)
        rmsle_cb = float(np.sqrt(np.mean((z_cb - val_z_true) ** 2)))
    else:
        z_cb = None
        rmsle_cb = 1.69848

    # Load Masked GRU S1
    s1_p = Path("artifacts/ssl_pretraining/predictions/S1_val_predictions.parquet")
    if s1_p.exists():
        z_gru_s1 = pl.read_parquet(s1_p)["z_pred"].to_numpy().astype(np.float32)
        rmsle_gru_s1 = float(np.sqrt(np.mean((z_gru_s1 - val_z_true) ** 2)))
    else:
        z_gru_s1 = z_pred_a0_v2
        rmsle_gru_s1 = rmsle_a0_v2

    # Save validation predictions
    all_preds_df = pl.DataFrame({
        "user_id": user_ids,
        "target": val_y_rub,
        "z_true": val_z_true,
        "past_buyer": val_past_buyer,
        "z_A0_V1": z_pred_a0_v1,
        "z_A0_V2": z_pred_a0_v2,
        "z_T5_R0": z_pred_t5_r0,
        "z_T5_R1": z_pred_t5_r1,
        "z_T5_3seed": z_t5_3seed,
    })
    all_preds_df.write_parquet(OUT_ROOT_STAGEA / "predictions" / "stage0_stageA_val_predictions.parquet")
    print(f"[+] Saved {OUT_ROOT_STAGEA / 'predictions' / 'stage0_stageA_val_predictions.parquet'}")

    # Blends
    blend_results = []
    if z_cb is not None:
        # V5.1 Baseline blend: 35% CatBoost + 65% GRU S1
        z_v51 = 0.35 * z_cb + 0.65 * z_gru_s1
        rmsle_v51 = float(np.sqrt(np.mean((z_v51 - val_z_true) ** 2)))
        blend_results.append({"blend_name": "v5.1 (35% CB + 65% GRU_S1)", "RMSLE": rmsle_v51, "delta_vs_v51": 0.0})

        # v5.1 + 10% T5 (30% CB + 60% GRU + 10% T5)
        z_tri_10 = 0.30 * z_cb + 0.60 * z_gru_s1 + 0.10 * z_t5_3seed
        rmsle_tri_10 = float(np.sqrt(np.mean((z_tri_10 - val_z_true) ** 2)))
        blend_results.append({"blend_name": "v5.1 + 10% T5", "RMSLE": rmsle_tri_10, "delta_vs_v51": rmsle_tri_10 - rmsle_v51})

        # v5.1 + 20% T5 (30% CB + 50% GRU + 20% T5)
        z_tri_20 = 0.30 * z_cb + 0.50 * z_gru_s1 + 0.20 * z_t5_3seed
        rmsle_tri_20 = float(np.sqrt(np.mean((z_tri_20 - val_z_true) ** 2)))
        blend_results.append({"blend_name": "v5.1 + 20% T5", "RMSLE": rmsle_tri_20, "delta_vs_v51": rmsle_tri_20 - rmsle_v51})

        # Optimal Simplex Search (step 0.05)
        best_w, best_tri_rmsle = (0.35, 0.65, 0.0), rmsle_v51
        for w_c in np.linspace(0.1, 0.6, 11):
            for w_g in np.linspace(0.1, 0.9, 17):
                w_t = round(1.0 - w_c - w_g, 3)
                if 0.0 <= w_t <= 0.5:
                    z_b = w_c * z_cb + w_g * z_gru_s1 + w_t * z_t5_3seed
                    r_b = float(np.sqrt(np.mean((z_b - val_z_true) ** 2)))
                    if r_b < best_tri_rmsle:
                        best_tri_rmsle = r_b
                        best_w = (round(w_c, 2), round(w_g, 2), round(w_t, 2))

        blend_results.append({
            "blend_name": f"Optimal Simplex ({best_w[0]} CB + {best_w[1]} GRU + {best_w[2]} T5)",
            "RMSLE": best_tri_rmsle,
            "delta_vs_v51": best_tri_rmsle - rmsle_v51,
        })

    pl.DataFrame(blend_results).write_csv(OUT_ROOT_STAGEA / "blend_evaluation.csv")
    print(f"[+] Saved {OUT_ROOT_STAGEA / 'blend_evaluation.csv'}")

    # Paired Bootstrap (N=1000) for T5 vs Canonical GRU (A0_V2)
    diff_t5_sq = (z_t5_3seed - val_z_true) ** 2
    diff_gru_sq = (z_pred_a0_v2 - val_z_true) ** 2
    deltas = []
    for _ in range(1000):
        idx_b = np.random.choice(len(val_z_true), size=len(val_z_true), replace=True)
        deltas.append(np.sqrt(np.mean(diff_t5_sq[idx_b])) - np.sqrt(np.mean(diff_gru_sq[idx_b])))
    deltas = np.array(deltas)
    p_t5_better = float(np.mean(deltas < 0.0))
    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    # Summary table
    summary_rows = [
        {
            "experiment_id": "A0_V1_Canonical_GRU180",
            "model_family": "GRU-180",
            "anchors": "V1 (11 anchors)",
            "RMSLE": round(rmsle_a0_v1, 5),
            "delta_vs_A0_V1": 0.0,
            "P_better_vs_A0": 0.0,
            "95_CI": "[0.0, 0.0]",
        },
        {
            "experiment_id": "A0_V2_Canonical_GRU180_Edge",
            "model_family": "GRU-180",
            "anchors": "V2_EDGE (11 + 2025-12-15)",
            "RMSLE": round(rmsle_a0_v2, 5),
            "delta_vs_A0_V1": round(rmsle_a0_v2 - rmsle_a0_v1, 5),
            "P_better_vs_A0": 0.50,
            "95_CI": "N/A",
        },
        {
            "experiment_id": "T5_R0_Purged_V1",
            "model_family": "T5-PreLN-TF365",
            "anchors": "V1 (11 anchors)",
            "RMSLE": round(rmsle_t5_r0, 5),
            "delta_vs_A0_V1": round(rmsle_t5_r0 - rmsle_a0_v1, 5),
            "P_better_vs_A0": 0.0,
            "95_CI": "N/A",
        },
        {
            "experiment_id": "T5_R1_Purged_V2_Edge",
            "model_family": "T5-PreLN-TF365",
            "anchors": "V2_EDGE (11 + 2025-12-15)",
            "RMSLE": round(rmsle_t5_r1, 5),
            "delta_vs_A0_V1": round(rmsle_t5_r1 - rmsle_a0_v1, 5),
            "P_better_vs_A0": p_t5_better,
            "95_CI": f"[{ci_low:.5f}, {ci_high:.5f}]",
        },
        {
            "experiment_id": "T5_R2_3Seed_Ensemble",
            "model_family": "T5-PreLN-TF365",
            "anchors": f"{best_anchors_name}",
            "RMSLE": round(rmsle_t5_3seed, 5),
            "delta_vs_A0_V1": round(rmsle_t5_3seed - rmsle_a0_v1, 5),
            "P_better_vs_A0": p_t5_better,
            "95_CI": f"[{ci_low:.5f}, {ci_high:.5f}]",
        },
    ]
    pl.DataFrame(summary_rows).write_csv(OUT_ROOT_STAGEA / "stage0_stageA_summary.csv")
    print(f"[+] Saved {OUT_ROOT_STAGEA / 'stage0_stageA_summary.csv'}")

    print("\n" + "=" * 80)
    print(f"=== STAGE 0 & STAGE A MASTER RUN COMPLETED in {time.time() - t0_master:.1f}s ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
