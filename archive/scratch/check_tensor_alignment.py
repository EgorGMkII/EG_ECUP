"""Check user_id alignment in seq_tensor."""

from pathlib import Path
import numpy as np
import polars as pl
from src.sequential.dataset import CACHE_DIR
from src.snapshots import get_or_create_selected_users

def main():
    u_selected = get_or_create_selected_users()
    print(f"selected_users_100k length: {len(u_selected):,}, first 5: {u_selected[:5]}")

    # Check snapshots
    snap_p = Path("data/snapshots/snapshot_2026-01-14.parquet")
    if snap_p.exists():
        snap = pl.read_parquet(snap_p)
        print(f"Snapshot 2026-01-14 length: {snap.height:,}, first 5 users: {snap['user_id'].to_list()[:5]}")
        print(f"Are snapshot users equal to selected_users? {snap['user_id'].to_list() == u_selected}")

    # Check cached tensors
    t_files = list(CACHE_DIR.glob("seq_tensor_*_u100000_t365.npy"))
    print(f"Found {len(t_files)} 100k tensors: {[f.name for f in t_files]}")

if __name__ == "__main__":
    main()
