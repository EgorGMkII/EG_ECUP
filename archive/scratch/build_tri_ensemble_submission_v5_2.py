"""Rock-Solid Tri-Ensemble Submission v5.2: CatBoost Transitions (40%) + MultiTask GRU (40%) + Patch Transformer-365 (20%)."""

import gc
import time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRegressor
from torch.utils.data import DataLoader, Dataset

from src.hurdle import get_feature_columns
from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.losses import MultiTaskHurdleLoss
from src.sequential.models import MultiTaskGRUModel, PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler
from src.sequential.trainer import train_multitask_gru
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.boosting import train_churn_classifier, train_reactivation_classifier
from src.transitions.features import compute_all_transition_features

TEST_ANCHOR = date(2026, 2, 13)
SUBMISSION_PATH = Path("data/submission.csv")
TRANSITIONS_ARTIFACTS = Path("artifacts/transitions")
TRANSITIONS_ARTIFACTS.mkdir(parents=True, exist_ok=True)


class PatchTransformerHurdleModel(nn.Module):
    """Patch Transformer with dedicated Classification and Conditional Buyer Regression Heads."""

    def __init__(
        self,
        input_dim: int = 15,
        patch_size: int = 7,
        num_patches: int = 52,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model

        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        # Temporal Attention pooling
        self.attn_linear = nn.Linear(d_model, 1)

        # Multi-task heads: Classification (All users) and Conditional Regression (Buyers only)
        self.head_cls = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_reg = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        x_trim = x[:, -self.num_patches * self.patch_size :, :]
        B = x_trim.shape[0]
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])

        tokens = self.patch_proj(x_patches) + self.pos_embed
        enc_out = self.transformer(tokens)

        # Attention pooling
        weights = torch.softmax(self.attn_linear(enc_out), dim=1)
        emb = torch.sum(weights * enc_out, dim=1)

        p_logits = self.head_cls(emb).squeeze(-1)
        z_cond = self.head_reg(emb).squeeze(-1)
        return p_logits, z_cond, emb


class MemmapMultiTask365Dataset(Dataset):
    """Multi-Anchor 365-day dataset streaming from disk with Buyer mask for Hurdle Loss."""

    def __init__(self, tensor_paths: list, targets_list: list, scaler=None):
        self.mmaps = [np.load(p, mmap_mode="r") for p in tensor_paths]
        self.lengths = [len(m) for m in self.mmaps]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.scaler = scaler

        y_all = np.concatenate(targets_list).astype(np.float32)
        self.y_true = torch.from_numpy(y_all).float()
        self.y_log = torch.log1p(torch.clamp(self.y_true, min=0.0))
        self.is_buyer = (self.y_true > 0.0).float()

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        tensor_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[tensor_idx]

        raw_seq = self.mmaps[tensor_idx][local_idx] # (365, 15)
        if self.scaler is not None:
            scaled_seq = (raw_seq - self.scaler.mean) / self.scaler.std
        else:
            scaled_seq = raw_seq

        x_tensor = torch.from_numpy(scaled_seq.astype(np.float32)).float()
        return x_tensor, self.y_log[idx], self.is_buyer[idx], self.y_true[idx]


def train_patch_transformer(model, dataset, epochs=8, batch_size=512, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MultiTaskHurdleLoss(lambda_cls=0.5, lambda_reg=1.0)

    print(f"[*] Training Patch Transformer on {device} ({epochs} epochs, {len(dataset):,} samples)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y_log, is_buyer, _ in loader:
            x, y_log, is_buyer = x.to(device), y_log.to(device), is_buyer.to(device)
            optimizer.zero_grad()
            p_logits, z_cond, _ = model(x)
            loss, _ = criterion(p_logits, z_cond, is_buyer, y_log)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        print(f"  Patch-TF Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Loss: {total_loss/max(n_batches, 1):.4f}")


def main():
    print("===================================================================")
    print("=== BUILDING TRI-ENSEMBLE SUBMISSION V5.2 ===")
    print("=== (CatBoost 40% + MultiTask GRU 40% + Patch-TF 20%) ===")
    print("===================================================================")
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pl.read_parquet(TRAIN_PARQUET)
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)
    print(f"[*] Total Test Users: {n_test:,}")

    anchors = generate_panel_anchors()
    train_sample_users = get_or_create_selected_users()

    # -------------------------------------------------------------------------
    # PART 1: CATBOOST TRANSITIONS (A3)
    # -------------------------------------------------------------------------
    cb_cache_path = TRANSITIONS_ARTIFACTS / "catboost_test_pred_v5_1.npy"
    if cb_cache_path.exists():
        print(f"\n[+] Loading cached CatBoost test predictions from {cb_cache_path}...")
        z_catboost_final = np.load(cb_cache_path)
    else:
        raise FileNotFoundError(f"{cb_cache_path} not found! Please run v5.1 first.")

    pred_cb_rub = np.expm1(z_catboost_final)
    print(f"[+] CatBoost Ready | Mean rub: {np.mean(pred_cb_rub):.2f} | P50: {np.median(pred_cb_rub):.2f} | P99: {np.percentile(pred_cb_rub, 99):.2f}")

    # -------------------------------------------------------------------------
    # PART 2: PROVEN MULTITASK GRU-90
    # -------------------------------------------------------------------------
    gru_cache_path = TRANSITIONS_ARTIFACTS / "gru_test_pred_v5_1.npy"
    if gru_cache_path.exists():
        print(f"\n[+] Loading cached MultiTask GRU test predictions from {gru_cache_path}...")
        z_gru_final = np.load(gru_cache_path)
    else:
        print("\n[*] Computing MultiTask GRU predictions...")
        from src.sequential.dataset import MemmapMultiAnchorDataset
        test_tensor_path_90 = CACHE_DIR / f"seq_tensor_2026-02-13_u{len(test_users)}_t90.npy"
        recent_anchors = anchors[-14:]
        train_paths_90, train_targets_90 = [], []
        for a in recent_anchors:
            filename = f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t90.npy"
            p = CACHE_DIR / filename
            train_paths_90.append(p)
            train_targets_90.append(extract_anchor_targets(data, train_sample_users, a))

        sample_tensor_90 = np.load(train_paths_90[-1])
        scaler_90 = SequentialScaler().fit(sample_tensor_90)
        del sample_tensor_90

        gru_train_ds = MemmapMultiAnchorDataset(train_paths_90, train_targets_90, scaler=scaler_90)
        gru_model = MultiTaskGRUModel(input_dim=15, hidden_dim=128, num_layers=2)
        train_multitask_gru(gru_model, gru_train_ds, epochs=10, batch_size=512, lambda_cls=0.5, lambda_reg=1.0, verbose=True)
        gru_model.eval()

        test_tensor_raw = np.load(test_tensor_path_90)
        test_tensor_scaled = scaler_90.transform(test_tensor_raw)
        del test_tensor_raw

        p_gru_list, z_cond_gru_list = [], []
        with torch.no_grad():
            for i in range(0, len(test_tensor_scaled), 1024):
                batch_x = torch.from_numpy(test_tensor_scaled[i : i + 1024]).float().to(device)
                p_logits_t, z_cond_t, _ = gru_model(batch_x)
                p_gru_list.append(torch.sigmoid(p_logits_t).cpu().numpy())
                z_cond_gru_list.append(torch.clamp(z_cond_t, min=0.0).cpu().numpy())

        p_gru = np.concatenate(p_gru_list)
        z_cond_gru = np.concatenate(z_cond_gru_list)
        z_gru_final = (np.power(p_gru, 1.1) * z_cond_gru).astype(np.float32)
        np.save(gru_cache_path, z_gru_final)
        del test_tensor_scaled, gru_model, gru_train_ds
        gc.collect()

    pred_gru_rub = np.expm1(z_gru_final)
    print(f"[+] MultiTask GRU Ready | Mean rub: {np.mean(pred_gru_rub):.2f} | P50: {np.median(pred_gru_rub):.2f} | P99: {np.percentile(pred_gru_rub, 99):.2f}")

    # -------------------------------------------------------------------------
    # PART 3: PATCH TRANSFORMER-365 (TRAINED WITH MULTITASK HURDLE LOSS)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 3] PATCH TRANSFORMER-365 TRAINING & INFERENCE ---")
    print("-------------------------------------------------------------------")

    test_tensor_path_365 = CACHE_DIR / f"seq_tensor_2026-02-13_u{len(test_users)}_t365.npy"
    if not test_tensor_path_365.exists():
        _ = get_cached_sequence_tensor(data, test_users, TEST_ANCHOR, seq_len=365)

    X_test_365_raw = np.load(test_tensor_path_365, mmap_mode="r")
    scaler_365 = SequentialScaler().fit(X_test_365_raw[:25000])

    seq_train_anchors = anchors[-8:]
    train_paths_365, train_targets_365 = [], []
    for a in seq_train_anchors:
        t_path = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
        if not t_path.exists():
            _ = get_cached_sequence_tensor(data, train_sample_users, a, seq_len=365)
        train_paths_365.append(t_path)
        train_targets_365.append(extract_anchor_targets(data, train_sample_users, a))

    tf_train_ds = MemmapMultiTask365Dataset(train_paths_365, train_targets_365, scaler=scaler_365)
    tf_model = PatchTransformerHurdleModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3)
    train_patch_transformer(tf_model, tf_train_ds, epochs=8, batch_size=512, lr=1e-3, device=device)
    tf_model.eval()

    # Predict on 250k test users
    print("\n[*] Inferring 250,000 test users through Patch Transformer...")
    p_tf_list, z_cond_tf_list = [], []
    inf_bs = 1024
    with torch.no_grad():
        for i in range(0, n_test, inf_bs):
            raw_batch = X_test_365_raw[i : i + inf_bs]
            scaled_batch = (raw_batch - scaler_365.mean) / scaler_365.std
            xb = torch.from_numpy(scaled_batch.astype(np.float32)).float().to(device)

            p_logits_t, z_cond_t, _ = tf_model(xb)
            p_tf_list.append(torch.sigmoid(p_logits_t).cpu().numpy())
            z_cond_tf_list.append(torch.clamp(z_cond_t, min=0.0).cpu().numpy())

    p_tf = np.concatenate(p_tf_list)
    z_cond_tf = np.concatenate(z_cond_tf_list)
    z_tf_final = (np.power(p_tf, 1.1) * z_cond_tf).astype(np.float32)

    pred_tf_rub = np.expm1(z_tf_final)
    print(f"[+] Patch Transformer Ready | Mean rub: {np.mean(pred_tf_rub):.2f} | P50: {np.median(pred_tf_rub):.2f} | P99: {np.percentile(pred_tf_rub, 99):.2f}")
    del tf_model, tf_train_ds
    gc.collect()

    # -------------------------------------------------------------------------
    # PART 4: TRI-ENSEMBLE BLENDING (40% CB + 40% GRU + 20% TF)
    # -------------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("--- [PART 4] TRI-ENSEMBLE BLENDING (40% CB + 40% GRU + 20% TF) ---")
    print("-------------------------------------------------------------------")

    z_final = (0.40 * z_catboost_final + 0.40 * z_gru_final + 0.20 * z_tf_final).astype(np.float32)
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
    print(f"  - Mean:  {mean_val:.2f} руб. (Target: ~33-42 руб.)")
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

    print(f"\n[+] SUBMISSION V5.2 SUCCESSFULLY SAVED to {SUBMISSION_PATH.resolve()} in {elapsed/60:.2f} min!")
    print("===================================================================")


if __name__ == "__main__":
    main()
