"""Audit of User-Embedding Experiments: Target Overlap, Transition States, and Purged Re-evaluation."""

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl

print("=" * 80)
print("=== COMPREHENSIVE AUDIT OF USER EMBEDDING & TARGET WINDOW OVERLAPS ===")
print("=" * 80)

# 1. TARGET WINDOW OVERLAP AUDIT
VAL_ANCHOR = date(2026, 1, 14)
VAL_TARGET_START = VAL_ANCHOR + timedelta(days=1)
VAL_TARGET_END = VAL_ANCHOR + timedelta(days=30)

print(f"\n[*] Validation Anchor: {VAL_ANCHOR}")
print(f"[*] Validation Target Period (30d): {VAL_TARGET_START} to {VAL_TARGET_END}")

# The 13 training anchors used in recent_14
train_anchors = [
    date(2025, 7, 21),
    date(2025, 8, 4),
    date(2025, 8, 18),
    date(2025, 9, 1),
    date(2025, 9, 15),
    date(2025, 9, 29),
    date(2025, 10, 13),
    date(2025, 10, 27),
    date(2025, 11, 10),
    date(2025, 11, 24),
    date(2025, 12, 8),
    date(2025, 12, 22),
    date(2026, 1, 5),
]

print("\n[*] Auditing Target Overlaps for all 13 Training Anchors:")
print(f"{'Anchor Date':<12} | {'History Window (180d)':<23} | {'Target Window (30d)':<23} | {'Overlap Days with Val Target':<30}")
print("-" * 100)

purged_train_anchors = []
leaking_train_anchors = []

for a in train_anchors:
    h_start = a - timedelta(days=179)
    h_end = a
    t_start = a + timedelta(days=1)
    t_end = a + timedelta(days=30)
    
    # Overlap with [VAL_TARGET_START, VAL_TARGET_END]
    overlap_start = max(t_start, VAL_TARGET_START)
    overlap_end = min(t_end, VAL_TARGET_END)
    
    if overlap_start <= overlap_end:
        overlap_days = (overlap_end - overlap_start).days + 1
    else:
        overlap_days = 0
        
    status = f"{overlap_days} DAYS LEAKAGE!" if overlap_days > 0 else "0 (PURGED / CLEAN)"
    print(f"{str(a):<12} | {str(h_start)}..{str(h_end):<10} | {str(t_start)}..{str(t_end):<10} | {status:<30}")
    
    if overlap_days > 0:
        leaking_train_anchors.append((a, overlap_days))
    else:
        purged_train_anchors.append(a)

print("\n" + "=" * 80)
print(f"CRITICAL FINDING: {len(leaking_train_anchors)} out of 13 train anchors have direct target leakage with validation target!")
for a, days in leaking_train_anchors:
    print(f"  - Anchor {a}: target {a + timedelta(days=1)}..{a + timedelta(days=30)} overlaps by {days} days with validation target {VAL_TARGET_START}..{VAL_TARGET_END}")

# Last legal train anchor with 30d embargo
last_legal_anchor = VAL_ANCHOR - timedelta(days=30)
print(f"\n[*] Last strictly purged training anchor for validation anchor {VAL_ANCHOR}: {last_legal_anchor} (target ends {last_legal_anchor + timedelta(days=30)} == {VAL_ANCHOR})")

# 2. CHECK TRANSITION MATRIX DEFINITION
print("\n[*] Auditing Transition States on Validation Anchor 2026-01-14...")
val_snap = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
print(f"Columns in snapshot: {val_snap.columns}")

target = val_snap["target"].to_numpy()
fut_buyer = (target > 0).astype(int)

# Check different definitions of past buyer
for col in ["gmv_sum_30d", "orders_sum_30d", "searches_sum_30d", "cart_sum_30d"]:
    if col in val_snap.columns:
        past = (val_snap[col].to_numpy() > 0).astype(int)
        c00 = np.sum((past == 0) & (fut_buyer == 0))
        c01 = np.sum((past == 0) & (fut_buyer == 1))
        c10 = np.sum((past == 1) & (fut_buyer == 0))
        c11 = np.sum((past == 1) & (fut_buyer == 1))
        print(f"  Definition '{col} > 0': 0->0: {c00}, 0->1: {c01}, 1->0: {c10}, 1->1: {c11}")

# Check 180d activity definition if exists
past_180_gmv = None
if "gmv_sum_180d" in val_snap.columns:
    past = (val_snap["gmv_sum_180d"].to_numpy() > 0).astype(int)
    c00 = np.sum((past == 0) & (fut_buyer == 0))
    c01 = np.sum((past == 0) & (fut_buyer == 1))
    c10 = np.sum((past == 1) & (fut_buyer == 0))
    c11 = np.sum((past == 1) & (fut_buyer == 1))
    print(f"  Definition 'gmv_sum_180d > 0': 0->0: {c00}, 0->1: {c01}, 1->0: {c10}, 1->1: {c11}")
