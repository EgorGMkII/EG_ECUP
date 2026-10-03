import polars as pl
import numpy as np
from scripts.run_stage_c_event_time_experiments import compute_event_sequences_for_anchor

print("Reading train.parquet...")
df_raw = pl.read_parquet("data/train.parquet")
print("Reading users...")
users_df = pl.read_parquet("artifacts/selected_users_100k.parquet")
user_ids = users_df["user_id"].to_numpy()[:200]

print("Extracting event sequences...")
c, t, age, rank, mask, empty = compute_event_sequences_for_anchor(df_raw, "2026-01-14", user_ids, max_events=128)
print("SUCCESS! Shapes:", c.shape, t.shape, age.shape, rank.shape, mask.shape, empty.shape)
print("Empty users in 200 sample:", empty.sum())
