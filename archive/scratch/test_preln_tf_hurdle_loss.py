"""Test PreLN Transformer with pure MultiTaskHurdleLoss (BCE on all + MSE on buyers, NO zero loss_dir)."""

import gc
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.losses import MultiTaskHurdleLoss
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET

TEST_ANCHOR = date(2026, 2, 13)


class PreLNTransformerHurdleModel(nn.Module):
    """2-Layer PreLN Transformer trained with Pure Hurdle Loss (No Zero-Loss Distortions)."""

    def __init__(self, input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-pooling: last patch (128) + mean patch (128) -> 256
        self.fusion = nn.Sequential(nn.Linear(d_model * 2, 128), nn.GELU())

        # Multi-task transition heads
        self.head_react = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x_trim = x[:, -self.num_patches * self.patch_size :, :]
        B = x_trim.shape[0]
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])
        tokens = self.patch_proj(x_patches) + self.pos_embed
        enc_out = self.transformer(tokens)

        last_t = enc_out[:, -1, :]
        mean_t = torch.mean(enc_out, dim=1)
        emb = self.fusion(torch.cat([last_t, mean_t], dim=-1))

        lr = self.head_react(emb).squeeze(-1)
        lc = self.head_churn(emb).squeeze(-1)
        zc = self.head_cond(emb).squeeze(-1)
        return lr, lc, zc, emb


class MultiAnchorHurdleDataset(Dataset):
    def __init__(self, tensor_paths, targets_list, past_buyer_list, scaler=None):
        self.mmaps = [np.load(p, mmap_mode="r") for p in tensor_paths]
        self.lengths = [len(m) for m in self.mmaps]
        self.cum_len = np.cumsum([0] + self.lengths)
        self.scaler = scaler

        y_all = np.concatenate(targets_list).astype(np.float32)
        past_b_all = np.concatenate(past_buyer_list).astype(np.float32)

        self.y_true = torch.from_numpy(y_all).float()
        self.y_log = torch.log1p(torch.clamp(self.y_true, min=0.0))
        self.past_buyer = torch.from_numpy(past_b_all).float()
        self.fut_buyer = (self.y_true > 0.0).float()

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, idx):
        t_idx = np.searchsorted(self.cum_len, idx, side="right") - 1
        l_idx = idx - self.cum_len[t_idx]
        raw_seq = self.mmaps[t_idx][l_idx]
        if self.scaler is not None:
            sc_seq = (raw_seq - self.scaler.mean) / self.scaler.std
        else:
            sc_seq = raw_seq
        return torch.from_numpy(sc_seq.astype(np.float32)).float(), self.y_log[idx], self.past_buyer[idx], self.fut_buyer[idx]


def train_preln_tf_hurdle(model, loader, epochs=6, lr=1e-3, device=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    print(f"[*] Training PreLN Transformer with Pure Hurdle Loss on {device} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y_log, past_b, fut_b in loader:
            x, y_log, past_b, fut_b = x.to(device), y_log.to(device), past_b.to(device), fut_b.to(device)
            optimizer.zero_grad()
            lr, lc, zc, _ = model(x)

            # 1. Reactivation Loss (Dormant users: past_b == 0)
            mask_dormant = past_b == 0
            if mask_dormant.sum() > 0:
                loss_react = bce_fn(lr[mask_dormant], fut_b[mask_dormant])
            else:
                loss_react = torch.tensor(0.0, device=device)

            # 2. Churn Loss (Active users: past_b == 1)
            mask_active = past_b == 1
            if mask_active.sum() > 0:
                loss_churn = bce_fn(lc[mask_active], 1.0 - fut_b[mask_active])
            else:
                loss_churn = torch.tensor(0.0, device=device)

            # 3. Conditional Regression Loss (STRICTLY on Active Buyers y_log > 0)
            mask_buyers = fut_b > 0.5
            if mask_buyers.sum() > 0:
                loss_cond = mse_fn(zc[mask_buyers], y_log[mask_buyers])
            else:
                loss_cond = torch.tensor(0.0, device=device)

            # Pure Hurdle Loss (NO direct zero penalty pulling encoder to 0)
            loss = 0.5 * loss_react + 0.5 * loss_churn + 1.0 * loss_cond
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | Loss: {total_loss/max(n_batches, 1):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)
    anchors = generate_panel_anchors()

    test_tensor_path_365 = CACHE_DIR / f"seq_tensor_2026-02-13_u{len(test_users)}_t365.npy"
    X_test_365_raw = np.load(test_tensor_path_365, mmap_mode="r")
    scaler_365 = SequentialScaler().fit(X_test_365_raw[:25000])

    # Test past buyer
    test_base_snap = build_snapshot(data, test_users, TEST_ANCHOR, is_test=True)
    past_gmv_test = test_base_snap["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_test = (past_gmv_test > 0).astype(np.int32)
    del test_base_snap

    # Training anchors
    seq_train_anchors = anchors[-8:]
    train_paths_365, y_tr_seq_list, past_b_tr_seq_list = [], [], []
    for a in seq_train_anchors:
        t_path = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
        train_paths_365.append(t_path)
        y_tr_seq_list.append(extract_anchor_targets(data, train_sample_users, a))
        snap_a = pl.read_parquet(f"data/snapshots/snapshot_{a.strftime('%Y-%m-%d')}.parquet")
        past_b_tr_seq_list.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
        del snap_a

    train_ds = MultiAnchorHurdleDataset(train_paths_365, y_tr_seq_list, past_b_tr_seq_list, scaler=scaler_365)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    model = PreLNTransformerHurdleModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    train_preln_tf_hurdle(model, train_loader, epochs=6, lr=1e-3, device=device)
    model.eval()

    # Predict on test
    print("\n[*] Inferring 250,000 test users through Pure Hurdle PreLN Transformer...")
    p_react_list, p_churn_list, z_cond_list = [], [], []
    inf_bs = 1024
    with torch.no_grad():
        for i in range(0, n_test, inf_bs):
            raw_batch = X_test_365_raw[i : i + inf_bs]
            sc_batch = (raw_batch - scaler_365.mean) / scaler_365.std
            xb = torch.from_numpy(sc_batch.astype(np.float32)).to(device)

            lr, lc, zc, _ = model(xb)
            p_react_list.append(torch.sigmoid(lr).cpu().numpy())
            p_churn_list.append(torch.sigmoid(lc).cpu().numpy())
            z_cond_list.append(zc.cpu().numpy())

    p_react = np.concatenate(p_react_list)
    p_churn = np.concatenate(p_churn_list)
    z_cond = np.concatenate(z_cond_list)

    p_buy = np.where(past_buyer_test == 0, p_react, 1.0 - p_churn)
    z_tf = (np.power(p_buy, 1.10) * z_cond).astype(np.float32)
    pred_rub = np.expm1(z_tf)

    print("\n[+] Pure Hurdle PreLN Transformer Test Quantiles:")
    print(f"    - Min:  {np.min(pred_rub):.2f} руб.")
    print(f"    - P50:  {np.median(pred_rub):.2f} руб. (Target: ~6-8 руб.)")
    print(f"    - Mean: {np.mean(pred_rub):.2f} руб. (Target: ~35-40 руб.)")
    print(f"    - P90:  {np.percentile(pred_rub, 90):.2f} руб. (Target: ~90-110 руб.)")
    print(f"    - P99:  {np.percentile(pred_rub, 99):.2f} руб. (Target: ~420-480 руб.)")
    print(f"    - Max:  {np.max(pred_rub):.2f} руб.")

    # Save cached test prediction
    np.save("artifacts/transitions/preln_transformer_test_pred_v6.npy", z_tf)
    print("[+] Saved to artifacts/transitions/preln_transformer_test_pred_v6.npy")

    # Combine Tri-Ensemble (35% CB + 45% GRU + 20% PreLN TF)
    z_cb = np.load("artifacts/transitions/catboost_test_pred_v5_1.npy")
    z_gru = np.load("artifacts/transitions/gru_test_pred_v5_1.npy")

    z_tri = (0.35 * z_cb + 0.45 * z_gru + 0.20 * z_tf).astype(np.float32)
    y_pred_tri = np.clip(np.expm1(z_tri), 0.0, None)

    print("\n[GUARDRAIL AUDIT FOR TRI-ENSEMBLE V6.0]:")
    print(f"    - Count: {len(y_pred_tri):,}")
    print(f"    - P50:   {np.median(y_pred_tri):.2f} руб.")
    print(f"    - Mean:  {np.mean(y_pred_tri):.2f} руб.")
    print(f"    - P90:   {np.percentile(y_pred_tri, 90):.2f} руб.")
    print(f"    - P99:   {np.percentile(y_pred_tri, 99):.2f} руб.")
    print(f"    - Max:   {np.max(y_pred_tri):.2f} руб.")

    pl.DataFrame({"user_id": test_users, "predict": y_pred_tri}).write_csv("data/submission.csv")
    pl.DataFrame({"user_id": test_users, "predict": y_pred_tri}).write_csv("data/submission_v6_tri_ensemble.csv")
    print("[+] SUBMISSION V6.0 SAVED TO data/submission.csv!")


if __name__ == "__main__":
    main()
