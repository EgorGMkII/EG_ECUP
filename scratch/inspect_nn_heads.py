"""Inspect NN and CatBoost Heads distributions on test."""

from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.sequential.dataset import CACHE_DIR
from src.sequential.models import HierarchicalGRUModel, PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler

def main():
    print("===================================================================")
    print("=== INSPECTING RAW HEAD OUTPUTS OF MODELS ===")
    print("===================================================================")

    test_tensor_path = CACHE_DIR / "seq_tensor_2026-02-13_u250000_t365.npy"
    if not test_tensor_path.exists():
        print("Tensor not found.")
        return

    X_test_365_raw = np.load(test_tensor_path, mmap_mode="r")
    scaler_365 = SequentialScaler().fit(X_test_365_raw[:25000])

    print(f"Test raw sequence mean: {np.mean(X_test_365_raw[:1000]):.4f}, std: {np.std(X_test_365_raw[:1000]):.4f}")
    print(f"Scaler mean: {scaler_365.mean[:3]}, std: {scaler_365.std[:3]}")

if __name__ == "__main__":
    main()
