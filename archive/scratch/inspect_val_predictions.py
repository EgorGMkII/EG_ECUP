"""Inspect validation predictions from experiment_long_seq_predictions.parquet."""

import polars as pl
import numpy as np
from pathlib import Path

def main():
    p = Path("artifacts/transitions/experiment_long_seq_predictions.parquet")
    df = pl.read_parquet(p)
    print("Columns:", df.columns)

    for col in ["z_gru90_fact", "z_gru365_fact", "z_tf365_fact", "z_hier_fact", "z_tf365_dir"]:
        z = df[col].to_numpy()
        rub = np.expm1(z)
        print(f"\n{col}:")
        print(f"  P10: {np.percentile(rub, 10):.2f}, P50: {np.median(rub):.2f}, Mean: {np.mean(rub):.2f}, P90: {np.percentile(rub, 90):.2f}, P99: {np.percentile(rub, 99):.2f}, Max: {np.max(rub):.2f}")

if __name__ == "__main__":
    main()
