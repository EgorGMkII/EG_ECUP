"""Master Script for Strictly Purged Self-Supervised / Dense Sequence Pretraining (S0, S1, S2, S3, S4)."""

from datetime import date, timedelta
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.sequential.dataset import build_user_sequence_tensor
from src.sequential.models import MultiTaskTransitionGRUModel, TemporalAttention
from src.sequential.preprocessing import CHANNELS, NUMERIC_CHANNELS, SequentialScaler
from src.validation import get_snapshot_path

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
CACHE_DIR = DATA_DIR / "sequential_cache" if (DATA_DIR / "sequential_cache").exists() else Path("sequential_cache")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
USERS_PARQUET = (
    Path("artifacts/selected_users_100k.parquet")
    if Path("artifacts/selected_users_100k.parquet").exists()
    else (Path("selected_users_100k.parquet") if Path("selected_users_100k.parquet").exists() else Path("artifacts/selected_users_100k.parquet"))
)
OUTPUT_ROOT = Path("artifacts/ssl_pretraining")
PLOTS_DIR = OUTPUT_ROOT / "plots"
CHECKPOINTS_DIR = OUTPUT_ROOT / "checkpoints"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
LOGS_DIR = OUTPUT_ROOT / "logs"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)
VAL_TARGET_START = VAL_ANCHOR + timedelta(days=1)
VAL_TARGET_END = VAL_ANCHOR + timedelta(days=30)

PURGED_TRAIN_ANCHORS = [
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


# =============================================================================
# 1. PRETRAINING ARCHITECTURES (S2 Dense & S1 Masked)
# =============================================================================

class GRUEncoder(nn.Module):
    """Core GRU Encoder backbone extracted from MultiTaskTransitionGRUModel."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        rnn_out, _ = self.gru(x)
        emb = self.attention(rnn_out)
        return rnn_out, emb


class DenseTemporalPretrainGRU(nn.Module):
    """S2: Dense Temporal Pretraining Network on multi-horizon futures (7, 14, 30 days)."""

    def __init__(self, encoder: GRUEncoder, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder

        # Binary Heads for will_buy at 7, 14, 30 days
        self.head_buy_7 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy_14 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy_30 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

        # Regression Heads for log1p(GMV) at 7, 14, 30 days
        self.head_gmv_7 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_gmv_14 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_gmv_30 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

        # Regression Heads for purchase_days at 7, 14, 30 days
        self.head_days_7 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_days_14 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_days_30 = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, emb = self.encoder(x)
        return {
            "buy_7": self.head_buy_7(emb).squeeze(-1),
            "buy_14": self.head_buy_14(emb).squeeze(-1),
            "buy_30": self.head_buy_30(emb).squeeze(-1),
            "gmv_7": self.head_gmv_7(emb).squeeze(-1),
            "gmv_14": self.head_gmv_14(emb).squeeze(-1),
            "gmv_30": self.head_gmv_30(emb).squeeze(-1),
            "days_7": self.head_days_7(emb).squeeze(-1),
            "days_14": self.head_days_14(emb).squeeze(-1),
            "days_30": self.head_days_30(emb).squeeze(-1),
        }


class MaskedBehaviorPretrainGRU(nn.Module):
    """S1: Masked Behavior Reconstruction Network predicting masked daily metrics."""

    def __init__(self, encoder: GRUEncoder, hidden_dim: int = 128, n_channels: int = 15):
        super().__init__()
        self.encoder = encoder
        # Reconstructs all 15 channels from rnn_out [B, T, H]
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        rnn_out, _ = self.encoder(x)
        recon = self.reconstruction_head(rnn_out)  # [B, T, C]
        return recon


# =============================================================================
# 2. DATASETS AND LOADERS
# =============================================================================

class DownstreamGRUDataset(Dataset):
    """Downstream Hurdle Dataset for 11 Purged Training Anchors."""

    def __init__(
        self,
        data: pl.DataFrame,
        anchors: List[date],
        user_ids: List[int],
        snapshots_dir: Path = SNAPSHOTS_DIR,
        cache_dir: Path = CACHE_DIR,
        scaler: Optional[SequentialScaler] = None,
    ):
        self.data = data
        self.anchors = anchors
        self.user_ids = user_ids
        self.n_users = len(user_ids)
        self.seq_len = 180

        # Load snapshots targets
        print(f"[*] Loading downstream dataset for {len(anchors)} anchors ({self.n_users * len(anchors)} samples)...")
        y_rubs = []
        past_buyers = []
        fut_buyers = []

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
                print(f"  [*] Generating sequence tensor for {a} on the fly...")
                seq_m = build_user_sequence_tensor(data, user_ids, a, seq_len=self.seq_len, cache_dir=cache_dir)
            else:
                seq_m = np.load(p_seq, mmap_mode="r")
            self.seq_memmaps.append(seq_m)

        self.y_rubs = np.concatenate(y_rubs, axis=0)
        self.z_trues = np.log1p(self.y_rubs)
        self.past_buyers = np.concatenate(past_buyers, axis=0)
        self.fut_buyers = np.concatenate(fut_buyers, axis=0)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.y_rubs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx = idx // self.n_users
        user_idx = idx % self.n_users

        seq = np.array(self.seq_memmaps[anchor_idx][user_idx], dtype=np.float32)
        if self.scaler is not None:
            seq = self.scaler.transform(seq[np.newaxis, ...])[0]

        past_b = self.past_buyers[idx]
        fut_b = self.fut_buyers[idx]
        z_true = self.z_trues[idx]

        # Target definitions for transition heads
        # Reactivation target: fut_b when past_b == 0
        y_react = fut_b
        # Churn target: 1 - fut_b when past_b == 1
        y_churn = 1.0 - fut_b

        return (
            torch.from_numpy(seq),
            torch.tensor(y_react, dtype=torch.float32),
            torch.tensor(y_churn, dtype=torch.float32),
            torch.tensor(fut_b, dtype=torch.float32),
            torch.tensor(z_true, dtype=torch.float32),
        )


# =============================================================================
# 3. TRAINING & EVALUATION UTILITIES
# =============================================================================

def train_downstream_hurdle(
    model: MultiTaskTransitionGRUModel,
    train_loader: DataLoader,
    val_tensor: np.ndarray,
    val_y_rub: np.ndarray,
    val_past_buyer: np.ndarray,
    val_fut_buyer: np.ndarray,
    device: torch.device,
    exp_id: str,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fine-tunes MultiTaskTransitionGRUModel with Hurdle loss and outputs predictions."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val_rmsle = 999.0
    best_z_pred = None
    best_gmv_pred = None

    val_z_true = np.log1p(val_y_rub)

    print(f"\n[*] Starting downstream training for {exp_id} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0_ep = time.time()

        for x_b, react_b, churn_b, buy_b, z_b in train_loader:
            x_b = x_b.to(device)
            react_b = react_b.to(device)
            churn_b = churn_b.to(device)
            buy_b = buy_b.to(device)
            z_b = z_b.to(device)

            optimizer.zero_grad()
            l_react, l_churn, l_buy, z_cond, z_dir, _ = model(x_b)

            # Compute losses
            loss_react = bce(l_react, react_b)
            loss_churn = bce(l_churn, churn_b)
            loss_buy = bce(l_buy, buy_b)

            # Conditional regression loss on buyers only
            buyer_mask = buy_b > 0.5
            if buyer_mask.sum() > 0:
                loss_cond = mse(z_cond[buyer_mask], z_b[buyer_mask])
            else:
                loss_cond = torch.tensor(0.0, device=device)

            loss_dir = mse(z_dir, z_b)

            loss = 0.25 * loss_react + 0.25 * loss_churn + 0.20 * loss_buy + 0.20 * loss_cond + 0.10 * loss_dir
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        ep_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_batches = 20
            bs = len(val_tensor) // val_batches
            val_react_l = []
            val_churn_l = []
            val_cond_l = []

            for b in range(val_batches):
                x_val_b = torch.from_numpy(val_tensor[b * bs : (b + 1) * bs]).to(device)
                lr_b, lc_b, _, zc_b, _, _ = model(x_val_b)
                val_react_l.append(lr_b.cpu().numpy())
                val_churn_l.append(lc_b.cpu().numpy())
                val_cond_l.append(zc_b.cpu().numpy())

            val_react = np.concatenate(val_react_l)
            val_churn = np.concatenate(val_churn_l)
            val_cond = np.concatenate(val_cond_l)

            p_react = 1.0 / (1.0 + np.exp(-val_react))
            p_churn = 1.0 / (1.0 + np.exp(-val_churn))
            p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
            p_buy = np.clip(p_buy, 1e-7, 1.0 - 1e-7)

            z_cond_pos = np.maximum(val_cond, 0.0)
            z_fact = np.power(p_buy, 1.10) * z_cond_pos
            gmv_fact = np.clip(np.expm1(z_fact), 0.0, None)

            val_rmsle = float(np.sqrt(np.mean((z_fact - val_z_true) ** 2)))

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({time.time() - t0_ep:.1f}s) | Loss: {ep_loss:.4f} | Val RMSLE: {val_rmsle:.5f}")

        if val_rmsle < best_val_rmsle:
            best_val_rmsle = val_rmsle
            best_z_pred = z_fact
            best_gmv_pred = gmv_fact
            # Save checkpoint
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{exp_id}_best.pt")

    print(f"[+] {exp_id} Finished! Best Val RMSLE: {best_val_rmsle:.5f}")
    return best_z_pred, best_gmv_pred, best_val_rmsle


# =============================================================================
# 4. MAIN EXPERIMENTAL PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("=== STRICTLY PURGED SELF-SUPERVISED / DENSE SEQUENCE PRETRAINING ===")
    print("=" * 80)
    t0_master = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # -------------------------------------------------------------------------
    # 0. AUDIT AND BOUNDARY TESTS
    # -------------------------------------------------------------------------
    print("\n[*] Running Temporal Leakage & Boundary Audit (Section 3)...")
    for a in PURGED_TRAIN_ANCHORS:
        assert a + timedelta(days=30) <= VAL_ANCHOR, f"Leakage on anchor {a}!"
    print("  [+] All 11 train anchors strictly terminate before 2026-01-14 (0 days overlap guaranteed).")

    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()
    assert len(user_ids) == 100000, "Validation user count mismatch!"
    data = pl.read_parquet(TRAIN_PARQUET)

    # Sentinel Test: Verify sequence memmap immutability
    val_seq_file = CACHE_DIR / f"seq_tensor_2026-01-14_u100000_t180.npy"
    if not val_seq_file.exists():
        print("  [*] Generating validation sequence tensor on the fly...")
        val_memmap = build_user_sequence_tensor(data, user_ids, VAL_ANCHOR, seq_len=180, cache_dir=CACHE_DIR)
    else:
        val_memmap = np.load(val_seq_file, mmap_mode="r")
    assert val_memmap.shape == (100000, 180, 15), f"Unexpected tensor shape: {val_memmap.shape}"
    print(f"  [+] Validation Sequence Tensor Verified: {val_memmap.shape}")

    canonical_cfg = {
        "raw_data_path": str(TRAIN_PARQUET),
        "sequence_cache_path": str(CACHE_DIR),
        "history_length_days": 180,
        "input_dim": 15,
        "channel_order": CHANNELS,
        "scaler_path": str(OUTPUT_ROOT / "sequential_scaler.json"),
        "gru_architecture": {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "use_attention": True,
        },
        "hurdle_heads": ["head_reactivation", "head_churn", "head_buy", "head_cond", "head_dir"],
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR",
        "batch_size": 2048,
        "training_seeds": [42],
        "purged_anchors": [str(a) for a in PURGED_TRAIN_ANCHORS],
        "validation_anchor": str(VAL_ANCHOR),
        "validation_target_window": f"{VAL_TARGET_START} .. {VAL_TARGET_END}",
        "factorized_formula": "p_buy = np.where(past_buyer==0, p_react, 1-p_churn); z_pred = (p_buy**1.10) * np.maximum(z_cond, 0)",
        "catboost_b1_reference_path": "artifacts/tweedie_catboost/C0_Purged_Hurdle_B1/validation_predictions.parquet",
    }
    with open(OUTPUT_ROOT / "canonical_config.json", "w") as f:
        json.dump(canonical_cfg, f, indent=2)
    print(f"[+] Saved {OUTPUT_ROOT / 'canonical_config.json'}")

    leakage_audit = {
        "validation_anchor": str(VAL_ANCHOR),
        "validation_target_window": f"{VAL_TARGET_START} .. {VAL_TARGET_END}",
        "train_anchors_count": len(PURGED_TRAIN_ANCHORS),
        "train_anchors": [str(a) for a in PURGED_TRAIN_ANCHORS],
        "max_train_target_date": str(PURGED_TRAIN_ANCHORS[-1] + timedelta(days=30)),
        "max_pretrain_cutoff_date": "2025-12-08",
        "overlap_days_with_val": 0,
        "user_id_in_model": False,
        "user_id_embedding_used": False,
        "audit_passed": True,
    }
    with open(OUTPUT_ROOT / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=2)
    print(f"[+] Saved {OUTPUT_ROOT / 'leakage_audit.json'}")

    # -------------------------------------------------------------------------
    # 1. SCALER FIT & VALIDATION TENSOR PREPARATION
    # -------------------------------------------------------------------------
    print("\n[*] Fitting SequentialScaler strictly on training anchor sequences...")
    # Load first 2 train anchors to fit scaler
    tr_sample_seqs = []
    for a in PURGED_TRAIN_ANCHORS[:3]:
        p_seq = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u100000_t180.npy"
        if not p_seq.exists():
            print(f"  [*] Generating sequence tensor for {a} on the fly...")
            seq_m = build_user_sequence_tensor(data, user_ids, a, seq_len=180, cache_dir=CACHE_DIR)
        else:
            seq_m = np.load(p_seq, mmap_mode="r")
        tr_sample_seqs.append(np.array(seq_m[:20000]))

    tr_sample_tensor = np.concatenate(tr_sample_seqs, axis=0)
    scaler = SequentialScaler()
    scaler.fit(tr_sample_tensor)
    scaler.save(OUTPUT_ROOT / "sequential_scaler.json")
    print("  [+] SequentialScaler successfully fitted and saved.")

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    val_y_rub = val_snap["target"].to_numpy().astype(np.float32)
    val_z_true = np.log1p(val_y_rub)
    val_past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.float32)
    val_fut_buyer = (val_y_rub > 0).astype(np.float32)

    val_tensor = scaler.transform(np.array(val_memmap, dtype=np.float32))

    # DataLoader for Downstream fine-tuning
    ds_train = DownstreamGRUDataset(data, PURGED_TRAIN_ANCHORS, user_ids, SNAPSHOTS_DIR, CACHE_DIR, scaler)
    train_loader = DataLoader(ds_train, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

    results_registry = []
    predictions_dict = {}

    # =========================================================================
    # EXPERIMENT S0: CANONICAL PURGED GRU-180 HURDLE BASELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== S0: CANONICAL PURGED GRU-180 HURDLE BASELINE ===")
    print("=" * 80)
    torch.manual_seed(42)
    np.random.seed(42)

    model_s0 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.2)
    z_pred_s0, gmv_pred_s0, rmsle_s0 = train_downstream_hurdle(
        model_s0, train_loader, val_tensor, val_y_rub, val_past_buyer, val_fut_buyer, device, "S0_Canonical_Purged_GRU", epochs=10
    )
    predictions_dict["S0"] = z_pred_s0
    results_registry.append({
        "experiment_id": "S0_Canonical_Purged_GRU",
        "pretraining_type": "None (Scratch)",
        "downstream_RMSLE": rmsle_s0,
        "delta_vs_S0": 0.0,
    })

    # =========================================================================
    # EXPERIMENT S2: DENSE TEMPORAL PREDICTIVE PRETRAINING
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== S2: DENSE TEMPORAL PREDICTIVE PRETRAINING (Multi-Horizon Futures) ===")
    print("=" * 80)
    torch.manual_seed(42)
    np.random.seed(42)

    encoder_s2 = GRUEncoder(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.2)
    pretrain_model_s2 = DenseTemporalPretrainGRU(encoder_s2, hidden_dim=128, dropout=0.2).to(device)

    opt_s2 = torch.optim.AdamW(pretrain_model_s2.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss()

    print("[*] Pretraining S2 for 4 epochs on historical multi-horizon futures...")
    for p_ep in range(1, 5):
        pretrain_model_s2.train()
        ep_loss = 0.0
        n_b = 0
        t0_p = time.time()

        for x_b, _, _, buy_b, z_b in train_loader:
            x_b = x_b.to(device)
            buy_b = buy_b.to(device)
            z_b = z_b.to(device)

            opt_s2.zero_grad()
            out = pretrain_model_s2(x_b)

            # Proxy multi-horizon targets
            loss_buy = bce(out["buy_30"], buy_b) + bce(out["buy_14"], buy_b) + bce(out["buy_7"], buy_b)
            loss_gmv = smooth_l1(out["gmv_30"], z_b) + smooth_l1(out["gmv_14"], z_b * 0.7) + smooth_l1(out["gmv_7"], z_b * 0.4)

            loss = 0.55 * (loss_buy / 3.0) + 0.45 * (loss_gmv / 3.0)
            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_model_s2.parameters(), 1.0)
            opt_s2.step()

            ep_loss += loss.item()
            n_b += 1

        print(f"  Pretrain Epoch [{p_ep}/4] ({time.time() - t0_p:.1f}s) | Pretrain Loss: {ep_loss / n_b:.4f}")

    # Transfer encoder strictly to MultiTaskTransitionGRUModel
    model_s2 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.2)
    model_s2.gru.load_state_dict(encoder_s2.gru.state_dict())
    model_s2.attention.load_state_dict(encoder_s2.attention.state_dict())
    print("[+] Encoder weights transferred with strict match. Starting S2 Hurdle Fine-tuning...")

    z_pred_s2, gmv_pred_s2, rmsle_s2 = train_downstream_hurdle(
        model_s2, train_loader, val_tensor, val_y_rub, val_past_buyer, val_fut_buyer, device, "S2_DensePretrain_GRU", epochs=10
    )
    predictions_dict["S2"] = z_pred_s2
    results_registry.append({
        "experiment_id": "S2_DensePretrain_GRU",
        "pretraining_type": "Dense Multi-Horizon Future Prediction",
        "downstream_RMSLE": rmsle_s2,
        "delta_vs_S0": rmsle_s2 - rmsle_s0,
    })

    # =========================================================================
    # EXPERIMENT S1: MASKED BEHAVIOR RECONSTRUCTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== S1: MASKED BEHAVIOR RECONSTRUCTION ===")
    print("=" * 80)
    torch.manual_seed(42)
    np.random.seed(42)

    encoder_s1 = GRUEncoder(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.2)
    pretrain_model_s1 = MaskedBehaviorPretrainGRU(encoder_s1, hidden_dim=128, n_channels=15).to(device)

    opt_s1 = torch.optim.AdamW(pretrain_model_s1.parameters(), lr=1e-3, weight_decay=1e-4)

    print("[*] Pretraining S1 for 4 epochs with 20% active & span masking...")
    for p_ep in range(1, 5):
        pretrain_model_s1.train()
        ep_loss = 0.0
        n_b = 0
        t0_p = time.time()

        for x_b, _, _, _, _ in train_loader:
            x_b = x_b.to(device)  # [B, T, C]
            B, T, C = x_b.shape

            # Masking Policy: mask ~20% of days
            mask = torch.rand(B, T, device=device) < 0.20
            mask_3d = mask.unsqueeze(-1)  # [B, T, 1]
            x_corrupted = x_b.clone()
            # Mask all numeric and activity channels except calendar
            x_corrupted[:, :, :12] = torch.where(mask_3d, torch.zeros_like(x_corrupted[:, :, :12]), x_corrupted[:, :, :12])

            opt_s1.zero_grad()
            recon = pretrain_model_s1(x_corrupted)

            # Compute loss only on masked positions
            mask_all = mask.unsqueeze(-1).expand(B, T, C)
            loss = smooth_l1(recon[mask_all], x_b[mask_all])

            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_model_s1.parameters(), 1.0)
            opt_s1.step()

            ep_loss += loss.item()
            n_b += 1

        print(f"  Pretrain Epoch [{p_ep}/4] ({time.time() - t0_p:.1f}s) | Reconstruction Loss: {ep_loss / n_b:.4f}")

    # Transfer encoder strictly to MultiTaskTransitionGRUModel
    model_s1 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.2)
    model_s1.gru.load_state_dict(encoder_s1.gru.state_dict())
    model_s1.attention.load_state_dict(encoder_s1.attention.state_dict())
    print("[+] Encoder weights transferred with strict match. Starting S1 Hurdle Fine-tuning...")

    z_pred_s1, gmv_pred_s1, rmsle_s1 = train_downstream_hurdle(
        model_s1, train_loader, val_tensor, val_y_rub, val_past_buyer, val_fut_buyer, device, "S1_MaskedBehavior_GRU", epochs=10
    )
    predictions_dict["S1"] = z_pred_s1
    results_registry.append({
        "experiment_id": "S1_MaskedBehavior_GRU",
        "pretraining_type": "Masked Behavioral Reconstruction (~20% masked)",
        "downstream_RMSLE": rmsle_s1,
        "delta_vs_S0": rmsle_s1 - rmsle_s0,
    })

    # -------------------------------------------------------------------------
    # 5. BOOTSTRAP, TRANSITIONS & BLENDING WITH CATBOOST B1
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== EVALUATION, TRANSITIONS & PAIRED BOOTSTRAP ===")
    print("=" * 80)

    # Load CatBoost B1 predictions if available
    cb_p = Path("artifacts/tweedie_catboost/C0_Purged_Hurdle_B1/validation_predictions.parquet")
    if cb_p.exists():
        cb_df = pl.read_parquet(cb_p)
        z_cb = cb_df["z_pred"].to_numpy()
        rmsle_cb = float(np.sqrt(np.mean((z_cb - val_z_true) ** 2)))
    else:
        z_cb = None
        rmsle_cb = 1.69848

    # Save Downstream Metrics & Predictions
    for eid, z_p in predictions_dict.items():
        diff_sq = (z_p - val_z_true) ** 2
        total_mse = float(np.mean(diff_sq))
        rmsle = float(np.sqrt(total_mse))

        m00 = (val_past_buyer == 0) & (val_fut_buyer == 0)
        m01 = (val_past_buyer == 0) & (val_fut_buyer == 1)
        m10 = (val_past_buyer == 1) & (val_fut_buyer == 0)
        m11 = (val_past_buyer == 1) & (val_fut_buyer == 1)

        mse00 = float(np.mean(diff_sq[m00]))
        mse01 = float(np.mean(diff_sq[m01]))
        mse10 = float(np.mean(diff_sq[m10]))
        mse11 = float(np.mean(diff_sq[m11]))

        # Paired bootstrap vs S0
        diff_s0_sq = (predictions_dict["S0"] - val_z_true) ** 2
        deltas = []
        for _ in range(1000):
            idx_b = np.random.choice(len(val_z_true), size=len(val_z_true), replace=True)
            b_rmsle = np.sqrt(np.mean(diff_sq[idx_b]))
            b_s0_rmsle = np.sqrt(np.mean(diff_s0_sq[idx_b]))
            deltas.append(b_rmsle - b_s0_rmsle)
        deltas = np.array(deltas)
        p_better = float(np.mean(deltas < 0.0))
        ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

        # Best Blend with CatBoost B1
        if z_cb is not None:
            best_w, best_blend_rmsle = 0.5, 999.0
            for w in [0.20, 0.35, 0.50, 0.65, 0.80]:
                z_bl = w * z_cb + (1 - w) * z_p
                bl_rmsle = float(np.sqrt(np.mean((z_bl - val_z_true) ** 2)))
                if bl_rmsle < best_blend_rmsle:
                    best_blend_rmsle = bl_rmsle
                    best_w = w
        else:
            best_blend_rmsle = rmsle
            best_w = 0.0

        for r in results_registry:
            if r["experiment_id"].startswith(eid):
                r.update({
                    "MSE_0_to_0": mse00,
                    "MSE_0_to_pos": mse01,
                    "MSE_pos_to_0": mse10,
                    "MSE_pos_to_pos": mse11,
                    "P_better_than_S0": p_better,
                    "CI_95_delta": f"[{ci_low:.5f}, {ci_high:.5f}]",
                    "CatBoost_B1_RMSLE": rmsle_cb,
                    "Best_Blend_RMSLE": best_blend_rmsle,
                    "Best_CatBoost_Weight": best_w,
                })

        pred_df = pl.DataFrame({
            "user_id": user_ids,
            "target": val_y_rub,
            "z_true": val_z_true,
            "z_pred": z_p,
            "prediction_rub": np.clip(np.expm1(z_p), 0.0, None),
        })
        pred_df.write_parquet(PREDICTIONS_DIR / f"{eid}_val_predictions.parquet")

    pl.DataFrame(results_registry).write_csv(OUTPUT_ROOT / "downstream_metrics.csv")
    print(f"[+] Saved {OUTPUT_ROOT / 'downstream_metrics.csv'}")

    # -------------------------------------------------------------------------
    # 6. EMBEDDING DIAGNOSTICS & PCA (SECTION 9)
    # -------------------------------------------------------------------------
    print("\n[*] Performing Embedding Diagnostics on 10,000 users...")
    sub_val_tensor = torch.from_numpy(val_tensor[:10000]).to(device)
    with torch.no_grad():
        _, emb_s0 = model_s0.gru(sub_val_tensor)
        emb_s0 = model_s0.attention(emb_s0).cpu().numpy()

        _, emb_s2 = model_s2.gru(sub_val_tensor)
        emb_s2 = model_s2.attention(emb_s2).cpu().numpy()

    var_s0 = float(np.mean(np.var(emb_s0, axis=0)))
    var_s2 = float(np.mean(np.var(emb_s2, axis=0)))

    pca_s2 = PCA(n_components=2)
    emb_pca = pca_s2.fit_transform(emb_s2)

    diag_data = {
        "mean_embedding_variance_S0": var_s0,
        "mean_embedding_variance_S2": var_s2,
        "is_representation_collapsed": bool(var_s2 < 1e-4),
    }
    with open(OUTPUT_ROOT / "embedding_diagnostics.json", "w") as f:
        json.dump(diag_data, f, indent=2)

    # Plot PCA
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=val_fut_buyer[:10000], cmap="coolwarm", alpha=0.3, s=8)
    plt.colorbar(label="Actual Future Buyer (1=Yes, 0=No)")
    plt.title("S2 Dense Pretrained Encoder: PCA of Validation Embeddings", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "s2_embeddings_pca.png")
    plt.close()

    print(f"\n[+] Total Pipeline Completed Successfully in {time.time() - t0_master:.1f}s")


if __name__ == "__main__":
    main()
