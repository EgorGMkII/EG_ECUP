"""Local smoke test for Stage 0 & Stage A (T5) components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from scripts.run_stage_0_and_a_t5_experiments import (
    T5SimplifiedTransformerModel,
    compute_behavioral_padding_mask
)

def test_t5_forward_backward():
    print("[*] Running T5 Forward & Backward smoke test...")
    B, T, C = 16, 365, 15
    x = torch.randn(B, T, C)
    # Some zero/dormant users
    x[0, :, :12] = 0.0
    x[1, :, :12] = 0.0
    x.requires_grad = True

    p_mask = compute_behavioral_padding_mask(x)
    assert p_mask[0].all() == True, "Dormant user should have all patches masked"
    assert p_mask[2].all() == False, "Active user should not have all patches masked"

    model = T5SimplifiedTransformerModel(input_dim=15, patch_size=7, num_patches=52, d_model=64, nhead=2, num_layers=2)
    lr, lc, lb, zc, zd, emb = model(x, padding_mask=p_mask)

    assert not torch.isnan(lr).any(), "NaN in reactivation logits!"
    assert not torch.isnan(zc).any(), "NaN in conditional z!"
    assert not torch.isnan(emb).any(), "NaN in embeddings!"
    assert lr.shape == (B,), f"Unexpected lr shape: {lr.shape}"
    assert zc.shape == (B,), f"Unexpected zc shape: {zc.shape}"
    assert emb.shape == (B, 64), f"Unexpected emb shape: {emb.shape}"

    loss = lr.sum() + zc.sum() + emb.sum()
    loss.backward()
    assert x.grad is not None, "Gradients failed to backprop!"
    print("  [+] T5 Forward, Backward, Padding Masking & NaN checks PASSED!")

if __name__ == "__main__":
    test_t5_forward_backward()
