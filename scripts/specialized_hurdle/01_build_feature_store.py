"""Script 01: Build & Audit Causal Feature Store across all 23 anchors."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import polars as pl
import yaml
from src.specialized_hurdle.feature_store import build_causal_feature_store


def main():
    print("=" * 80)
    print("01: BUILD & AUDIT CAUSAL FEATURE STORE")
    print("=" * 80)

    snapshots_dir = Path("data/snapshots")
    out_dir = Path("artifacts/specialized_hurdle/feature_store")
    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/specialized_hurdle/canonical_anchors.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    anchors = cfg["anchors"]

    users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()
    print(f"[*] Building Causal Feature Store for {len(anchors)} anchors (100k users)...")

    feature_cols, feature_hash = build_causal_feature_store(
        snapshots_dir=snapshots_dir,
        out_dir=out_dir,
        anchors=anchors,
        user_ids=users_100k,
    )

    print(f"\n[+] Feature store ready: {len(feature_cols)} features per anchor (Hash: {feature_hash}).")


if __name__ == "__main__":
    main()
