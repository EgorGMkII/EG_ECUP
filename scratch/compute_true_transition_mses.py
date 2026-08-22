"""Exact transition state MSE calculation on actual predictions."""
import polars as pl
import numpy as np

val_snap = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
y_true_log = np.log1p(val_snap["target"].to_numpy().astype(np.float64))
past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(int)
fut_buyer = (val_snap["target"].to_numpy() > 0).astype(int)

m00 = (past_buyer == 0) & (fut_buyer == 0)
m01 = (past_buyer == 0) & (fut_buyer == 1)
m10 = (past_buyer == 1) & (fut_buyer == 0)
m11 = (past_buyer == 1) & (fut_buyer == 1)

n00, n01, n10, n11 = np.sum(m00), np.sum(m01), np.sum(m10), np.sum(m11)
print(f"Correct Cohort Sizes: 0->0: {n00}, 0->1: {n01}, 1->0: {n10}, 1->1: {n11}, Total: {n00+n01+n10+n11}")

for var in ["E0_seed42", "E1_seed42", "E2_seed42", "E3_seed42", "E2_shuffled_seed42"]:
    p_df = pl.read_parquet(f"artifacts/user_embedding/{var}/validation_predictions.parquet")
    z_fact = p_df["factorized_z"].to_numpy().astype(np.float64)
    
    diff_sq = (z_fact - y_true_log) ** 2
    total_mse = float(np.mean(diff_sq))
    total_rmsle = float(np.sqrt(total_mse))
    
    mse00 = float(np.mean(diff_sq[m00]))
    mse01 = float(np.mean(diff_sq[m01]))
    mse10 = float(np.mean(diff_sq[m10]))
    mse11 = float(np.mean(diff_sq[m11]))
    
    weighted_mse = (n00 * mse00 + n01 * mse01 + n10 * mse10 + n11 * mse11) / 100000.0
    
    print(f"\n[{var}] Total MSE: {total_mse:.5f} (RMSLE: {total_rmsle:.5f}) | Weighted Sum MSE: {weighted_mse:.5f}")
    print(f"  0->0 ({n00}): {mse00:.5f}")
    print(f"  0->1 ({n01}): {mse01:.5f}")
    print(f"  1->0 ({n10}): {mse10:.5f}")
    print(f"  1->1 ({n11}): {mse11:.5f}")
    print(f"  Arithmetic check diff: {abs(total_mse - weighted_mse):.2e}")
