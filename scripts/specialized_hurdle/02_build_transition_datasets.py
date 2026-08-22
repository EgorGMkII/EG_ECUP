"""Script 02: Build Specialized Transition Datasets (Manifest, React, Churn, Amount)."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
from src.specialized_hurdle.data_builder import build_all_specialized_datasets
from src.specialized_hurdle.definitions import ALL_AVAILABLE_ANCHORS


def main():
    print("=" * 80)
    print("02: BUILD SPECIALIZED TRANSITION DATASETS")
    print("=" * 80)

    snapshots_dir = Path("data/snapshots")
    out_dir = Path("artifacts/specialized_hurdle/datasets")
    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    users_100k_path = Path("artifacts/selected_users_100k.parquet")
    user_ids = pl.read_parquet(users_100k_path)["user_id"].to_numpy() if users_100k_path.exists() else None
    print(f"[*] Building transition datasets across {len(ALL_AVAILABLE_ANCHORS)} anchors (100k users)...")

    df_manifest, df_react, df_churn, df_amount = build_all_specialized_datasets(
        snapshots_dir=snapshots_dir,
        out_dir=out_dir,
        anchors=ALL_AVAILABLE_ANCHORS,
        user_ids=user_ids,
    )

    print(f"[+] Total manifest rows: {len(df_manifest):,}")
    print(f"[+] Reactivation dataset rows (was_active == False): {len(df_react):,}")
    print(f"[+] Churn dataset rows (was_active == True): {len(df_churn):,}")
    print(f"[+] Amount dataset rows (future_gmv_30d > 0): {len(df_amount):,}")

    # Compute class balance per anchor
    balance_records = []
    for anchor in sorted(df_manifest["anchor"].unique()):
        sub = df_manifest.filter(pl.col("anchor") == anchor)
        n_tot = len(sub)
        n_0_0 = len(sub.filter(pl.col("transition_state") == "0->0"))
        n_0_1 = len(sub.filter(pl.col("transition_state") == "0->1"))
        n_1_0 = len(sub.filter(pl.col("transition_state") == "1->0"))
        n_1_1 = len(sub.filter(pl.col("transition_state") == "1->1"))

        react_pos_rate = n_0_1 / max(1, (n_0_0 + n_0_1))
        churn_pos_rate = n_1_0 / max(1, (n_1_0 + n_1_1))
        n_amount = n_0_1 + n_1_1

        balance_records.append({
            "anchor": anchor,
            "n_total": n_tot,
            "n_0_0": n_0_0,
            "n_0_1": n_0_1,
            "n_1_0": n_1_0,
            "n_1_1": n_1_1,
            "react_positive_rate": react_pos_rate,
            "churn_positive_rate": churn_pos_rate,
            "amount_sample_count": n_amount,
        })

    df_balance = pl.DataFrame(balance_records)
    balance_csv = reports_dir / "dataset_balance.csv"
    df_balance.write_csv(balance_csv)
    print(f"\n[+] Saved dataset balance report to {balance_csv}")
    print(df_balance)


if __name__ == "__main__":
    main()
