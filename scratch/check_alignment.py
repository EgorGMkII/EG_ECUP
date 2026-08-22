"""Check why transition counts differed."""
import polars as pl
import numpy as np

val_snap = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
pred_df = pl.read_parquet("artifacts/user_embedding/E0_seed42/validation_predictions.parquet")

print(f"val_snap user_ids (first 5): {val_snap['user_id'][:5].to_list()}")
print(f"pred_df user_ids (first 5): {pred_df['user_id'][:5].to_list()}")

# Check if user order is identical
is_same_order = (val_snap['user_id'].to_list() == pred_df['user_id'].to_list())
print(f"Are user_id orders identical? {is_same_order}")

# In pred_df:
past_pred = pred_df['current_state'].to_numpy()
fut_pred = (pred_df['z_true'].to_numpy() > 0).astype(int)

c00 = np.sum((past_pred == 0) & (fut_pred == 0))
c01 = np.sum((past_pred == 0) & (fut_pred == 1))
c10 = np.sum((past_pred == 1) & (fut_pred == 0))
c11 = np.sum((past_pred == 1) & (fut_pred == 1))
print(f"pred_df transitions: 0->0: {c00}, 0->1: {c01}, 1->0: {c10}, 1->1: {c11}")

# In val_snap:
past_snap = (val_snap['gmv_sum_30d'].to_numpy() > 0).astype(int)
fut_snap = (val_snap['target'].to_numpy() > 0).astype(int)
s00 = np.sum((past_snap == 0) & (fut_snap == 0))
s01 = np.sum((past_snap == 0) & (fut_snap == 1))
s10 = np.sum((past_snap == 1) & (fut_snap == 0))
s11 = np.sum((past_snap == 1) & (fut_snap == 1))
print(f"val_snap transitions: 0->0: {s00}, 0->1: {s01}, 1->0: {s10}, 1->1: {s11}")
