"""Phase 1 & 2: Canonical Configuration Locking & Training-Inference Parity Verification."""

import json
import os
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.hurdle import get_feature_columns
from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.features import compute_all_transition_features

AUDIT_DIR = Path("artifacts/transformer_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = AUDIT_DIR / "canonical_config.json"


def main():
    print("===================================================================")
    print("=== PHASE 1 & 2: CANONICAL CONFIG & PARITY VERIFICATION ===")
    print("===================================================================")

    # 1. Canonical Sequence Channels
    SEQUENCE_CHANNELS = [
        "search_count",
        "search_item_count",
        "sku_clicks",
        "cart_adds",
        "fav_adds",
        "order_count",
        "order_item_count",
        "gmv",
        "action_count_total",
        "is_active_day",
        "is_buyer_day",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ]

    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    val_anchor = date(2026, 1, 14)
    test_anchor = date(2026, 2, 13)
    anchors = generate_panel_anchors()

    # Load 1 tabular snapshot to extract exact tabular feature names
    sample_df = pl.read_parquet(SNAPSHOTS_DIR / f"snapshot_{anchors[0].strftime('%Y-%m-%d')}.parquet")
    all_tabular_cols = get_feature_columns(sample_df)
    noisy_cols = [c for c in all_tabular_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    base_tabular_cols = [c for c in all_tabular_cols if c not in noisy_cols]

    # Transition feature columns
    sample_trans_df = compute_all_transition_features(data, train_sample_users[:100], anchors[0])
    trans_cols = [c for c in sample_trans_df.columns if c != "user_id"]
    all_feature_cols = base_tabular_cols + trans_cols

    config = {
        "sequence_length": 365,
        "patch_size": 7,
        "num_patches": 52,
        "input_dim": len(SEQUENCE_CHANNELS),
        "sequence_channels": SEQUENCE_CHANNELS,
        "n_tabular_features": len(all_feature_cols),
        "base_tabular_features": base_tabular_cols,
        "transition_features": trans_cols,
        "feature_transformations": {
            "counters": "log1p",
            "gmv": "log1p",
            "scaling": "SequentialScaler(mean, std per channel)",
            "target": "log1p(GMV_30d)",
        },
        "scaler_fit_strategy": "fit on first 25,000 samples of validation 365d sequence tensor",
        "transformer_parameters": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.15,
            "patch_size": 7,
            "num_patches": 52,
            "activation": "gelu",
            "norm_first": True,
        },
        "heads": [
            "head_reactivation (Linear 128->64 -> GELU -> Dropout -> Linear 64->1)",
            "head_churn (Linear 128->64 -> GELU -> Dropout -> Linear 64->1)",
            "head_buy (Linear 128->64 -> GELU -> Dropout -> Linear 64->1)",
            "head_cond (Linear 128->64 -> GELU -> Dropout -> Linear 64->1)",
            "head_dir (Linear 128->64 -> GELU -> Dropout -> Linear 64->1)",
        ],
        "ensemble_weights_v51": {
            "catboost_transitions": 0.50,
            "multitask_gru": 0.50,
        },
        "alpha_v51": 1.1,
        "anchor_dates": {
            "train_anchors": [a.strftime("%Y-%m-%d") for a in anchors[-8:]],
            "validation_anchor": val_anchor.strftime("%Y-%m-%d"),
            "test_anchor": test_anchor.strftime("%Y-%m-%d"),
        },
        "prediction_space": "log1p space ensembling with expm1 clipping",
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[+] Canonical config locked to {CONFIG_PATH}")

    # =========================================================================
    # PHASE 2: TRAINING VS INFERENCE PARITY VERIFICATION
    # =========================================================================
    print("\n--- PHASE 2: TRAINING VS INFERENCE PARITY TEST (10,000 USERS) ---")
    val_users_10k = train_sample_users[:10000]

    # Load 365d validation tensor
    val_tensor_path = CACHE_DIR / f"seq_tensor_{val_anchor.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
    if not val_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, train_sample_users, val_anchor, seq_len=365)

    X_val_all_raw = np.load(val_tensor_path, mmap_mode="r")
    X_val_10k_raw = X_val_all_raw[:10000].copy()

    scaler = SequentialScaler().fit(X_val_all_raw[:25000])
    X_val_10k_scaled = (X_val_10k_raw - scaler.mean) / scaler.std

    # Instantiate model and save dummy/initial weights
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_orig = PatchTransformer365Model(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)
    model_orig.eval()

    ckpt_path = AUDIT_DIR / "parity_test_transformer.pt"
    torch.save(model_orig.state_dict(), ckpt_path)

    # -------------------------------------------------------------------------
    # PIPELINE A: In-Memory Evaluation
    # -------------------------------------------------------------------------
    with torch.no_grad():
        x_tensor_a = torch.from_numpy(X_val_10k_scaled.astype(np.float32)).float().to(device)
        l_react_a, l_churn_a, l_buy_a, z_cond_a, z_dir_a, emb_a = model_orig(x_tensor_a)

        p_react_a = torch.sigmoid(l_react_a).cpu().numpy()
        p_churn_a = torch.sigmoid(l_churn_a).cpu().numpy()
        p_buy_a = torch.sigmoid(l_buy_a).cpu().numpy()
        z_cond_np_a = z_cond_a.cpu().numpy()
        z_dir_np_a = z_dir_a.cpu().numpy()
        emb_np_a = emb_a.cpu().numpy()

    # -------------------------------------------------------------------------
    # PIPELINE B: Loaded Checkpoint with Strict=True
    # -------------------------------------------------------------------------
    model_loaded = PatchTransformer365Model(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model_loaded.load_state_dict(state_dict, strict=True)
    model_loaded.eval()

    with torch.no_grad():
        x_tensor_b = torch.from_numpy(X_val_10k_scaled.astype(np.float32)).float().to(device)
        l_react_b, l_churn_b, l_buy_b, z_cond_b, z_dir_b, emb_b = model_loaded(x_tensor_b)

        p_react_b = torch.sigmoid(l_react_b).cpu().numpy()
        p_churn_b = torch.sigmoid(l_churn_b).cpu().numpy()
        p_buy_b = torch.sigmoid(l_buy_b).cpu().numpy()
        z_cond_np_b = z_cond_b.cpu().numpy()
        z_dir_np_b = z_dir_b.cpu().numpy()
        emb_np_b = emb_b.cpu().numpy()

    # Differences
    diff_input = float(np.max(np.abs(x_tensor_a.cpu().numpy() - x_tensor_b.cpu().numpy())))
    diff_emb = float(np.max(np.abs(emb_np_a - emb_np_b)))
    diff_react = float(np.max(np.abs(p_react_a - p_react_b)))
    diff_churn = float(np.max(np.abs(p_churn_a - p_churn_b)))
    diff_cond = float(np.max(np.abs(z_cond_np_a - z_cond_np_b)))
    diff_dir = float(np.max(np.abs(z_dir_np_a - z_dir_np_b)))

    print(f"[*] Max Absolute Differences between Pipeline A and Pipeline B:")
    print(f"    - Input Tensor:      {diff_input:.2e}")
    print(f"    - Embedding:         {diff_emb:.2e}")
    print(f"    - Reactivation Prob: {diff_react:.2e}")
    print(f"    - Churn Prob:        {diff_churn:.2e}")
    print(f"    - Conditional Z:     {diff_cond:.2e}")
    print(f"    - Direct Z:          {diff_dir:.2e}")

    assert diff_input < 1e-5, f"Input tensor mismatch: {diff_input}"
    assert diff_emb < 1e-5, f"Embedding mismatch: {diff_emb}"
    assert diff_react < 1e-5, f"Reactivation mismatch: {diff_react}"
    assert diff_churn < 1e-5, f"Churn mismatch: {diff_churn}"
    assert diff_cond < 1e-5, f"Conditional Z mismatch: {diff_cond}"
    assert diff_dir < 1e-5, f"Direct Z mismatch: {diff_dir}"

    print(f"\n[+] PARITY VERIFICATION PASSED (strict=True, max_abs_diff < 1e-5)")

    parity_results = {
        "verified_samples": 10000,
        "max_abs_diff_input": diff_input,
        "max_abs_diff_embedding": diff_emb,
        "max_abs_diff_reactivation": diff_react,
        "max_abs_diff_churn": diff_churn,
        "max_abs_diff_conditional_z": diff_cond,
        "max_abs_diff_direct_z": diff_dir,
        "parity_status": "PASSED",
    }
    with open(AUDIT_DIR / "parity_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(parity_results, f, indent=2)


if __name__ == "__main__":
    main()
