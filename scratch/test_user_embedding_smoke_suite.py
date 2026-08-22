"""Comprehensive Smoke-Test Suite for User-Embedding GRU-180 (Mandatory Pre-Launch Checklist)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sequential.user_embedding import UserEmbeddingResidualGRU
from src.sequential.user_embedding_dataset import UserMemmapDataset
from src.sequential.preprocessing import SequentialScaler

print("=" * 80)
print("=== MANDATORY LOCAL SMOKE-TEST SUITE FOR USER-EMBEDDING GRU-180 ===")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. DATASET SMOKE TEST
# -----------------------------------------------------------------------------
print("\n[*] 1. Testing Dataset shapes, dtypes, and user_idx integrity...")
dummy_tensor = np.random.randn(256, 180, 15).astype(np.float32)
dummy_targets = np.random.exponential(scale=50.0, size=256).astype(np.float32)
dummy_past_b = (np.random.rand(256) > 0.5).astype(np.float32)
dummy_user_idx = np.random.randint(1, 250001, size=256, dtype=np.int64)

tensor_p = Path("artifacts/user_embedding/scratch_dummy_tensor.npy")
np.save(tensor_p, dummy_tensor)

ds = UserMemmapDataset(
    tensor_paths=[tensor_p],
    targets_list=[dummy_targets],
    past_buyer_list=[dummy_past_b],
    user_idx_list=[dummy_user_idx],
    seq_len=180,
    scaler=None,
    shuffle_user_idx=False,
)
loader = DataLoader(ds, batch_size=16, shuffle=False)
x_b, y_log_b, past_b, fut_b, u_idx_b = next(iter(loader))

assert x_b.shape == (16, 180, 15), f"Unexpected x_b shape: {x_b.shape}"
assert x_b.dtype == torch.float32, f"Unexpected x_b dtype: {x_b.dtype}"
assert u_idx_b.shape == (16,), f"Unexpected u_idx_b shape: {u_idx_b.shape}"
assert u_idx_b.dtype == torch.int64, f"Unexpected u_idx_b dtype: {u_idx_b.dtype}"
assert (u_idx_b >= 0).all() and (u_idx_b <= 250000).all(), "user_idx out of range [0, 250000]!"
print("[+] Dataset smoke-test PASSED (Shapes, dtypes, ranges 100% verified)!")

# -----------------------------------------------------------------------------
# 2. MODEL FORWARD PASS SHAPES (E0, E1, E2, E3)
# -----------------------------------------------------------------------------
print("\n[*] 2. Testing Model Forward Pass for E0, E1, E2, E3...")
for var in ["E0", "E1", "E2", "E3"]:
    model = UserEmbeddingResidualGRU(variant=var)
    model.eval()
    with torch.no_grad():
        lr, lc, lb, zc, zd, emb = model(x_b, u_idx_b)

    assert lr.shape == (16,), f"[{var}] Bad lr shape: {lr.shape}"
    assert lc.shape == (16,), f"[{var}] Bad lc shape: {lc.shape}"
    assert zc.shape == (16,), f"[{var}] Bad zc shape: {zc.shape}"
    assert emb.shape == (16, 128), f"[{var}] Bad emb shape: {emb.shape}"
    assert not torch.isnan(lr).any(), f"[{var}] NaN in lr!"
    assert not torch.isnan(zc).any(), f"[{var}] NaN in zc!"
    print(f"  [+] {var} Forward Pass verified (Outputs shape [16], Emb [16, 128])")

# -----------------------------------------------------------------------------
# 3. GRADIENT FLOW AND BACKWARD STEP
# -----------------------------------------------------------------------------
print("\n[*] 3. Testing Backward Step and Gradient Flow for E0, E1, E2, E3...")
bce_fn = nn.BCEWithLogitsLoss()
mse_fn = nn.MSELoss()

for var in ["E0", "E1", "E2", "E3"]:
    model = UserEmbeddingResidualGRU(variant=var)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    lr, lc, _, zc, _, _ = model(x_b, u_idx_b)
    loss = 0.5 * (bce_fn(lr, fut_b) + bce_fn(lc, 1.0 - fut_b)) + mse_fn(zc, y_log_b)
    loss.backward()

    # Verify GRU gradients exist
    assert model.gru.weight_ih_l0.grad is not None, f"[{var}] GRU grad is None!"
    assert not torch.isnan(model.gru.weight_ih_l0.grad).any(), f"[{var}] NaN in GRU grad!"

    # Verify User branch gradients
    if var == "E1":
        assert model.user_bias_react.weight.grad is not None, "E1 bias_react grad is None!"
        assert model.user_bias_churn.weight.grad is not None, "E1 bias_churn grad is None!"
    elif var in ["E2", "E3"]:
        assert model.user_embedding.weight.grad is not None, f"[{var}] embedding grad is None!"
        assert model.mlp_react[0].weight.grad is not None, f"[{var}] mlp_react grad is None!"
        assert model.mlp_churn[0].weight.grad is not None, f"[{var}] mlp_churn grad is None!"
        if var == "E3":
            assert model.mlp_cond[0].weight.grad is not None, "E3 mlp_cond grad is None!"

    optimizer.step()
    print(f"  [+] {var} Gradient Flow verified (No NaNs, all expected branches receive gradients)!")

# -----------------------------------------------------------------------------
# 4. CHECKPOINT ROUND-TRIP TEST
# -----------------------------------------------------------------------------
print("\n[*] 4. Testing Checkpoint Save / Load Round-Trip (strict=True)...")
for var in ["E0", "E1", "E2", "E3"]:
    m1 = UserEmbeddingResidualGRU(variant=var)
    m1.eval()
    with torch.no_grad():
        out1 = m1(x_b, u_idx_b)

    ckpt_path = Path(f"artifacts/user_embedding/scratch_roundtrip_{var}.pt")
    torch.save(m1.state_dict(), ckpt_path)

    m2 = UserEmbeddingResidualGRU(variant=var)
    m2.load_state_dict(torch.load(ckpt_path), strict=True)
    m2.eval()
    with torch.no_grad():
        out2 = m2(x_b, u_idx_b)

    max_diff = float(torch.max(torch.abs(out1[0] - out2[0])).item())
    assert max_diff < 1e-6, f"[{var}] Round-trip mismatch: {max_diff}"
    print(f"  [+] {var} Round-Trip PASSED (Max diff: {max_diff:.2e})")
    ckpt_path.unlink(missing_ok=True)

del ds, loader
import gc
gc.collect()

# -----------------------------------------------------------------------------
# 5. MINI-RUN (256 users, 1 epoch, 1 validation pass)
# -----------------------------------------------------------------------------
print("\n[*] 5. Testing Mini-Run (256 users, 1 epoch, validation pass)...")
mini_ds = UserMemmapDataset(
    tensor_paths=[tensor_p],
    targets_list=[dummy_targets],
    past_buyer_list=[dummy_past_b],
    user_idx_list=[dummy_user_idx],
    seq_len=180,
    scaler=None,
)
mini_loader = DataLoader(mini_ds, batch_size=64, shuffle=True)
m_mini = UserEmbeddingResidualGRU(variant="E2")
opt_mini = torch.optim.AdamW(m_mini.parameters(), lr=1e-3)

m_mini.train()
for xb, y_log_b, past_b, fut_b, u_idx_b in mini_loader:
    opt_mini.zero_grad()
    lr, lc, _, zc, _, _ = m_mini(xb, u_idx_b)
    loss = 0.5 * (bce_fn(lr, fut_b) + bce_fn(lc, 1.0 - fut_b)) + mse_fn(zc, y_log_b)
    loss.backward()
    opt_mini.step()

m_mini.eval()
with torch.no_grad():
    lr_v, lc_v, _, zc_v, _, _ = m_mini(xb, u_idx_b)
assert len(lr_v) == 64, "Mini validation pass failed!"
print("  [+] Mini-Run (E2, 256 users, 1 epoch) PASSED successfully!")

del mini_ds, mini_loader
gc.collect()
try:
    tensor_p.unlink(missing_ok=True)
except Exception:
    pass

print("\n" + "=" * 80)
print("=== ALL 5 MANDATORY SMOKE-TESTS PASSED 100% WITHOUT ERROR! ===")
print("=" * 80)
