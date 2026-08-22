"""Feature Store Builder for Specialized Hurdle Stack."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def build_causal_feature_store(
    snapshots_dir: Path,
    out_dir: Path,
    anchors: List[str],
    user_ids: Optional[np.ndarray] = None,
) -> Tuple[List[str], str]:
    """Extracts causal tabular feature sets for all anchors, ensuring exact schema parity."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base feature list from the canonical snapshot (excluding metadata/targets)
    ref_snap = pl.read_parquet(snapshots_dir / "snapshot_2026-01-14.parquet")
    excluded = {
        "user_id", "anchor_date", "history_start", "history_end",
        "target_start", "target_end", "target", "will_buy_30d", "user_segment_id"
    }
    feature_cols = [c for c in ref_snap.columns if c not in excluded]
    feature_hash = hashlib.sha256(json.dumps(feature_cols).encode()).hexdigest()[:16]

    print(f"[*] Causal Feature Store: {len(feature_cols)} features per anchor (Hash: {feature_hash})")

    for a in anchors:
        snap_file = snapshots_dir / f"snapshot_{a}.parquet"
        if not snap_file.exists():
            continue
        snap = pl.read_parquet(snap_file)
        if user_ids is not None:
            snap = snap.filter(pl.col("user_id").is_in(set(user_ids)))

        # Ensure user ordering
        u_map = {u: i for i, u in enumerate(snap["user_id"].to_list())}
        order = [u_map[u] for u in user_ids if u in u_map] if user_ids is not None else list(range(len(snap)))
        snap_ordered = snap[order]

        # Select user_id + features + targets without duplicates
        cols_to_save = ["user_id"] + [c for c in feature_cols if c in snap_ordered.columns]
        if "target" in snap_ordered.columns and "target" not in cols_to_save:
            cols_to_save.append("target")
        if "lifetime_gmv" in snap_ordered.columns and "lifetime_gmv" not in cols_to_save:
            cols_to_save.append("lifetime_gmv")

        cols_to_save = list(dict.fromkeys(cols_to_save))
        out_file = out_dir / f"anchor_{a}.parquet"
        snap_ordered.select(cols_to_save).write_parquet(out_file)
        print(f"   -> Saved {len(snap_ordered)} rows for anchor {a} to {out_file.name}")

    return feature_cols, feature_hash
