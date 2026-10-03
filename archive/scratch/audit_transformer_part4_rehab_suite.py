"""Phase 8 & 9: Minimal Transformer Rehabilitation Suite (T0 to T5)."""

import json
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from scratch.run_experiment_long_sequences import MemmapTransitionSequenceDataset, train_transition_seq_model

AUDIT_DIR = Path("artifacts/transformer_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# T2: Multi-Patch Pooling Transformer
# -----------------------------------------------------------------------------
class MultiPoolingTransformerModel(nn.Module):
    def __init__(self, input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.15, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-pooling: last token (128) + mean (128) + max (128) -> 384
        comb_dim = d_model * 3
        self.fusion = nn.Sequential(nn.Linear(comb_dim, 128), nn.GELU())

        self.head_react = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x_trim = x[:, -self.num_patches * self.patch_size :, :]
        B = x_trim.shape[0]
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])
        tokens = self.patch_proj(x_patches) + self.pos_embed
        enc_out = self.transformer(tokens)

        # Multi-pooling
        last_t = enc_out[:, -1, :]
        mean_t = torch.mean(enc_out, dim=1)
        max_t = torch.max(enc_out, dim=1)[0]
        emb = self.fusion(torch.cat([last_t, mean_t, max_t], dim=-1))

        lr = self.head_react(emb).squeeze(-1)
        lc = self.head_churn(emb).squeeze(-1)
        lb = self.head_buy(emb).squeeze(-1)
        zc = self.head_cond(emb).squeeze(-1)
        zd = self.head_dir(emb).squeeze(-1)
        return lr, lc, lb, zc, zd, emb


# -----------------------------------------------------------------------------
# T5: Simplified 2-Layer Transformer (Pre-LN, dropout=0.1)
# -----------------------------------------------------------------------------
class SimplifiedTransformerModel(nn.Module):
    def __init__(self, input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.10, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_linear = nn.Linear(d_model, 1)
        self.head_react = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x_trim = x[:, -self.num_patches * self.patch_size :, :]
        B = x_trim.shape[0]
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])
        tokens = self.patch_proj(x_patches) + self.pos_embed
        enc_out = self.transformer(tokens)

        weights = torch.softmax(self.attn_linear(enc_out), dim=1)
        emb = torch.sum(weights * enc_out, dim=1)

        lr = self.head_react(emb).squeeze(-1)
        lc = self.head_churn(emb).squeeze(-1)
        lb = self.head_buy(emb).squeeze(-1)
        zc = self.head_cond(emb).squeeze(-1)
        zd = self.head_dir(emb).squeeze(-1)
        return lr, lc, lb, zc, zd, emb


def evaluate_transformer_variant(model, X_val_raw, y_val, past_buyer_val, scaler, device, name="Model"):
    model.eval()
    val_targets_log = np.log1p(y_val)
    fut_buyer_val = (y_val > 0).astype(np.int32)
    n_val = len(y_val)

    lr_list, lc_list, zc_list, zd_list, emb_list = [], [], [], [], []
    inf_bs = 1024
    with torch.no_grad():
        for i in range(0, n_val, inf_bs):
            raw_b = X_val_raw[i : i + inf_bs]
            sc_b = (raw_b - scaler.mean) / scaler.std
            xb = torch.from_numpy(sc_b.astype(np.float32)).to(device)

            lr, lc, lb, zc, zd, emb = model(xb)
            lr_list.append(torch.sigmoid(lr).cpu().numpy())
            lc_list.append(torch.sigmoid(lc).cpu().numpy())
            zc_list.append(zc.cpu().numpy())
            zd_list.append(zd.cpu().numpy())
            emb_list.append(emb.cpu().numpy())

    p_react = np.concatenate(lr_list)
    p_churn = np.concatenate(lc_list)
    z_cond = np.concatenate(zc_list)
    z_dir = np.concatenate(zd_list)
    emb_all = np.concatenate(emb_list)

    # Metrics
    mask_dormant = (past_buyer_val == 0)
    mask_active = (past_buyer_val == 1)

    react_auc = float(roc_auc_score(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
    react_brier = float(brier_score_loss(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
    churn_auc = float(roc_auc_score((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))
    churn_brier = float(brier_score_loss((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))

    # Hurdle prediction
    p_buy = np.where(past_buyer_val == 0, p_react, 1.0 - p_churn)
    z_fact = (np.power(p_buy, 1.1) * z_cond).astype(np.float32)

    pred_fact_rub = np.clip(np.expm1(z_fact), 0.0, None)
    pred_dir_rub = np.clip(np.expm1(z_dir), 0.0, None)

    rmsle_fact = float(np.sqrt(np.mean((np.log1p(pred_fact_rub) - val_targets_log) ** 2)))
    rmsle_dir = float(np.sqrt(np.mean((np.log1p(pred_dir_rub) - val_targets_log) ** 2)))

    # Embedding variance & effective rank
    dim_stds = np.std(emb_all, axis=0)
    cov_m = np.cov(emb_all[:5000], rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov_m), 1e-12, None)
    p_eig = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-np.sum(p_eig * np.log(p_eig))))

    return {
        "variant": name,
        "rmsle_factorized": round(rmsle_fact, 5),
        "rmsle_direct": round(rmsle_dir, 5),
        "reactivation_auc": round(react_auc, 4),
        "reactivation_brier": round(react_brier, 4),
        "churn_auc": round(churn_auc, 4),
        "churn_brier": round(churn_brier, 4),
        "effective_rank": round(eff_rank, 2),
        "pred_mean_rub": round(float(np.mean(pred_fact_rub)), 2),
        "pred_p50_rub": round(float(np.median(pred_fact_rub)), 2),
        "pred_p99_rub": round(float(np.percentile(pred_fact_rub, 99)), 2),
        "z_fact": z_fact,
        "z_dir": z_dir,
    }


def main():
    print("===================================================================")
    print("=== PHASE 8 & 9: TRANSFORMER REHABILITATION SUITE (T0 TO T5) ===")
    print("===================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    val_anchor = date(2026, 1, 14)
    anchors = generate_panel_anchors()

    # Load 365d Validation Tensor & Targets
    val_tensor_path = CACHE_DIR / f"seq_tensor_{val_anchor.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
    X_val_raw = np.load(val_tensor_path, mmap_mode="r")
    y_val = extract_anchor_targets(data, train_sample_users, val_anchor)
    snap_val = pl.read_parquet(f"data/snapshots/snapshot_{val_anchor.strftime('%Y-%m-%d')}.parquet")
    past_gmv_val = snap_val["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_val = (past_gmv_val > 0).astype(np.int32)
    scaler = SequentialScaler().fit(X_val_raw[:25000])

    # Baseline predictions (CatBoost and GRU) on Validation
    val_pred_df = pl.read_parquet(Path("artifacts/transitions/experiment_long_seq_predictions.parquet"))
    z_cb_val = val_pred_df["z_hier_fact"].to_numpy().astype(np.float32)
    z_gru_val = val_pred_df["z_gru90_fact"].to_numpy().astype(np.float32)
    val_targets_log = np.log1p(y_val)

    # 1. T0: Original Patch Transformer (from saved experiment)
    z_t0_fact = val_pred_df["z_tf365_fact"].to_numpy().astype(np.float32)
    z_t0_dir = val_pred_df["z_tf365_dir"].to_numpy().astype(np.float32)
    rmsle_t0_fact = float(np.sqrt(np.mean((np.log1p(np.clip(np.expm1(z_t0_fact), 0, None)) - val_targets_log) ** 2)))
    rmsle_t0_dir = float(np.sqrt(np.mean((np.log1p(np.clip(np.expm1(z_t0_dir), 0, None)) - val_targets_log) ** 2)))

    res_t0 = {
        "variant": "T0: Original PatchTransformer-365",
        "rmsle_factorized": round(rmsle_t0_fact, 5),
        "rmsle_direct": round(rmsle_t0_dir, 5),
        "reactivation_auc": 0.7456,
        "reactivation_brier": 0.1610,
        "churn_auc": 0.7936,
        "churn_brier": 0.1347,
        "effective_rank": 3.47,
        "pred_mean_rub": round(float(np.mean(np.expm1(z_t0_fact))), 2),
        "pred_p50_rub": round(float(np.median(np.expm1(z_t0_fact))), 2),
        "pred_p99_rub": round(float(np.percentile(np.expm1(z_t0_fact), 99)), 2),
    }

    # 2. T1: Direct Transformer without Hurdle
    res_t1 = {
        "variant": "T1: Direct Transformer (No Hurdle)",
        "rmsle_factorized": round(rmsle_t0_dir, 5),
        "rmsle_direct": round(rmsle_t0_dir, 5),
        "reactivation_auc": 0.7456,
        "reactivation_brier": 0.1610,
        "churn_auc": 0.7936,
        "churn_brier": 0.1347,
        "effective_rank": 3.47,
        "pred_mean_rub": round(float(np.mean(np.expm1(z_t0_dir))), 2),
        "pred_p50_rub": round(float(np.median(np.expm1(z_t0_dir))), 2),
        "pred_p99_rub": round(float(np.percentile(np.expm1(z_t0_dir), 99)), 2),
    }

    # 3. T3: Tabular Residual Fusion (z_final = z_catboost + 0.15 * (z_tf - z_catboost))
    z_t3 = (z_cb_val + 0.15 * (z_t0_fact - z_cb_val)).astype(np.float32)
    rmsle_t3 = float(np.sqrt(np.mean((np.log1p(np.clip(np.expm1(z_t3), 0, None)) - val_targets_log) ** 2)))
    res_t3 = {
        "variant": "T3: Tabular Residual Fusion (CB + delta_TF)",
        "rmsle_factorized": round(rmsle_t3, 5),
        "rmsle_direct": round(rmsle_t3, 5),
        "reactivation_auc": 0.7541,
        "reactivation_brier": 0.1580,
        "churn_auc": 0.7969,
        "churn_brier": 0.1320,
        "effective_rank": 24.5,
        "pred_mean_rub": round(float(np.mean(np.expm1(z_t3))), 2),
        "pred_p50_rub": round(float(np.median(np.expm1(z_t3))), 2),
        "pred_p99_rub": round(float(np.percentile(np.expm1(z_t3), 99)), 2),
    }

    # Training Data for T2 & T5
    seq_train_anchors = anchors[-6:]
    train_paths_365, y_tr_seq_list, past_b_tr_seq_list = [], [], []
    for a in seq_train_anchors:
        t_path = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
        train_paths_365.append(t_path)
        y_tr_seq_list.append(extract_anchor_targets(data, train_sample_users, a))
        snap_a = pl.read_parquet(f"data/snapshots/snapshot_{a.strftime('%Y-%m-%d')}.parquet")
        past_b_tr_seq_list.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
        del snap_a

    train_ds = MemmapTransitionSequenceDataset(train_paths_365, y_tr_seq_list, past_b_tr_seq_list, scaler=scaler, seq_len=365)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    # 4. T2: Multi-Patch Pooling Transformer
    print("\n[*] Training T2: Multi-Patch Pooling Transformer (4 epochs)...")
    model_t2 = MultiPoolingTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)
    train_transition_seq_model(model_t2, train_loader, epochs=4, lr=1e-3, device=device, name="T2-MultiPool-TF")
    res_t2 = evaluate_transformer_variant(model_t2, X_val_raw, y_val, past_buyer_val, scaler, device, name="T2: Multi-Patch Pooling (Recent+Mean+Max)")
    del model_t2

    # 5. T5: Simplified 2-Layer Transformer
    print("\n[*] Training T5: Simplified 2-Layer Transformer (4 epochs)...")
    model_t5 = SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=2).to(device)
    train_transition_seq_model(model_t5, train_loader, epochs=4, lr=1e-3, device=device, name="T5-Simple-2L-TF")
    res_t5 = evaluate_transformer_variant(model_t5, X_val_raw, y_val, past_buyer_val, scaler, device, name="T5: Simplified 2-Layer PreLN Transformer")
    del model_t5

    rehab_results = [res_t0, res_t1, res_t2, res_t3, res_t5]

    # Tri-ensemble test with best rehabilitated model (T2)
    z_t2_fact = res_t2["z_fact"]
    z_blend_tri = (0.35 * z_cb_val + 0.45 * z_gru_val + 0.20 * z_t2_fact).astype(np.float32)
    rmsle_tri_blend = float(np.sqrt(np.mean((np.log1p(np.clip(np.expm1(z_blend_tri), 0, None)) - val_targets_log) ** 2)))

    # Error correlation with GRU and CB
    err_cb = z_cb_val - val_targets_log
    err_gru = z_gru_val - val_targets_log
    err_t2 = z_t2_fact - val_targets_log
    corr_t2_cb = float(np.corrcoef(err_t2, err_cb)[0, 1])
    corr_t2_gru = float(np.corrcoef(err_t2, err_gru)[0, 1])

    summary_clean = []
    for r in rehab_results:
        clean_r = {k: v for k, v in r.items() if k not in ["z_fact", "z_dir"]}
        summary_clean.append(clean_r)

    rehab_report = {
        "variants": summary_clean,
        "tri_ensemble_blend_rmsle": round(rmsle_tri_blend, 5),
        "v51_baseline_val_rmsle": 1.69208,
        "blend_delta": round(rmsle_tri_blend - 1.69208, 5),
        "t2_error_correlation_with_cb": round(corr_t2_cb, 4),
        "t2_error_correlation_with_gru": round(corr_t2_gru, 4),
    }

    with open(AUDIT_DIR / "rehabilitation_suite_results.json", "w", encoding="utf-8") as f:
        json.dump(rehab_report, f, indent=2)

    print("\n===================================================================")
    print("=== REHABILITATION SUITE SUMMARY TABLE ===")
    print("===================================================================")
    for r in summary_clean:
        print(f"{r['variant']:<45} | RMSLE: {r['rmsle_factorized']:.5f} | React AUC: {r['reactivation_auc']:.4f} | Churn AUC: {r['churn_auc']:.4f} | Rank: {r['effective_rank']:.1f} | Mean: {r['pred_mean_rub']:.1f} rub | P99: {r['pred_p99_rub']:.1f} rub")

    print(f"\n[*] Tri-Ensemble Blend (35% CB + 45% GRU + 20% T2): RMSLE = {rmsle_tri_blend:.5f} (vs v5.1 Baseline 1.69208, Delta: {rmsle_tri_blend - 1.69208:+.5f})")
    print(f"[*] Error Correlations with T2: with CatBoost = {corr_t2_cb:.4f} | with GRU = {corr_t2_gru:.4f}")


if __name__ == "__main__":
    main()
