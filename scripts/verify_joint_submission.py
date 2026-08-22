import hashlib
from pathlib import Path
import numpy as np
import polars as pl

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    old_sub_path = Path("submission_specialized_hurdle_stack.csv")
    new_sub_path = Path("submission_specialized_hurdle_joint_rmsle.csv")
    diag_path = Path("submission_specialized_hurdle_joint_rmsle_diagnostics.parquet")
    raw_path = Path("test_specialists_raw_predictions_250k.parquet")
    sample_sub_path = Path("sample_submit.csv")

    sha_old = get_sha256(old_sub_path) if old_sub_path.exists() else "N/A"
    sha_new = get_sha256(new_sub_path) if new_sub_path.exists() else "N/A"

    print("=" * 80)
    print("1. INTEGRITY AUDIT & CHECKSUMS")
    print("=" * 80)
    print(f"Old Submission ({old_sub_path.name}): SHA256 = {sha_old}")
    print(f"New Submission ({new_sub_path.name}): SHA256 = {sha_new}")
    print(f"Diagnostics:    {diag_path.name} ({diag_path.stat().st_size:,} bytes)")
    print(f"Raw 250k:       {raw_path.name} ({raw_path.stat().st_size:,} bytes)")

    df_old = pl.read_csv(old_sub_path)
    df_new = pl.read_csv(new_sub_path)
    df_diag = pl.read_parquet(diag_path)
    df_raw = pl.read_parquet(raw_path)
    df_sample = pl.read_csv(sample_sub_path)

    # 1. Integrity verification
    print("\n2. INTEGRITY VERIFICATION:")
    print(f"  Row count New vs Sample: {len(df_new):,} vs {len(df_sample):,} (Match: {len(df_new) == len(df_sample)})")
    users_match = (df_new["user_id"].to_numpy() == df_sample["user_id"].to_numpy()).all()
    print(f"  Exact user_id order matching sample_submit.csv: {users_match}")
    print(f"  Missing / NaN / Inf count: is_null={df_new['predict'].is_null().sum()}, is_nan={df_new['predict'].is_nan().sum()}")
    print(f"  Non-negativity: min_predict = {df_new['predict'].min():.6f} (>= 0: {df_new['predict'].min() >= 0})")

    # 2. Raw Predictions audit
    print("\n3. RAW TEST SPECIALIST PREDICTIONS (250,000 users):")
    for col in df_raw.columns:
        n_nan = df_raw[col].is_nan().sum() if df_raw[col].dtype in [pl.Float32, pl.Float64] else 0
        n_null = df_raw[col].is_null().sum()
        mean_val = df_raw[col].mean() if df_raw[col].dtype in [pl.Float32, pl.Float64] else 0.0
        print(f"  {col:<22}: null={n_null:<5} nan={n_nan:<5} mean={mean_val:.4f}")

    # 3. Distribution Metrics
    y_old = df_old["predict"].to_numpy()
    y_new = df_new["predict"].to_numpy()
    was_act = df_diag["was_active"].to_numpy()

    def calc_stats(arr):
        return {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "p01": np.percentile(arr, 1),
            "p10": np.percentile(arr, 10),
            "p50": np.percentile(arr, 50),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99),
            "max": np.max(arr),
            "share_lt_1": np.mean(arr < 1.0) * 100,
            "share_gt_100": np.mean(arr > 100.0) * 100,
            "share_gt_1000": np.mean(arr > 1000.0) * 100,
        }

    stats_all_old = calc_stats(y_old)
    stats_all_new = calc_stats(y_new)
    stats_inact_old = calc_stats(y_old[was_act == 0])
    stats_inact_new = calc_stats(y_new[was_act == 0])
    stats_act_old = calc_stats(y_old[was_act == 1])
    stats_act_new = calc_stats(y_new[was_act == 1])

    print("\n4. DISTRIBUTION COMPARISON (OVERALL):")
    print(f"  Mean:        Old = {stats_all_old['mean']:.2f} RUB | New = {stats_all_new['mean']:.2f} RUB")
    print(f"  Std:         Old = {stats_all_old['std']:.2f} RUB | New = {stats_all_new['std']:.2f} RUB")
    print(f"  Median(P50): Old = {stats_all_old['p50']:.2f} RUB | New = {stats_all_new['p50']:.2f} RUB")
    print(f"  P90:         Old = {stats_all_old['p90']:.2f} RUB | New = {stats_all_new['p90']:.2f} RUB")
    print(f"  P95:         Old = {stats_all_old['p95']:.2f} RUB | New = {stats_all_new['p95']:.2f} RUB")
    print(f"  P99:         Old = {stats_all_old['p99']:.2f} RUB | New = {stats_all_new['p99']:.2f} RUB")
    print(f"  Max:         Old = {stats_all_old['max']:.2f} RUB | New = {stats_all_new['max']:.2f} RUB")
    print(f"  Share <1 RUB:   Old = {stats_all_old['share_lt_1']:.2f}% | New = {stats_all_new['share_lt_1']:.2f}%")
    print(f"  Share >100 RUB: Old = {stats_all_old['share_gt_100']:.2f}% | New = {stats_all_new['share_gt_100']:.2f}%")
    print(f"  Share >1k RUB:  Old = {stats_all_old['share_gt_1000']:.2f}% | New = {stats_all_new['share_gt_1000']:.2f}%")

    print("\n5. INACTIVE USERS (was_active == 0, N=117,143):")
    print(f"  Mean:        Old = {stats_inact_old['mean']:.2f} RUB | New = {stats_inact_new['mean']:.2f} RUB")
    print(f"  Median(P50): Old = {stats_inact_old['p50']:.2f} RUB | New = {stats_inact_new['p50']:.2f} RUB")
    print(f"  P90:         Old = {stats_inact_old['p90']:.2f} RUB | New = {stats_inact_new['p90']:.2f} RUB")
    print(f"  P99:         Old = {stats_inact_old['p99']:.2f} RUB | New = {stats_inact_new['p99']:.2f} RUB")

    print("\n6. ACTIVE USERS (was_active == 1, N=132,857):")
    print(f"  Mean:        Old = {stats_act_old['mean']:.2f} RUB | New = {stats_act_new['mean']:.2f} RUB")
    print(f"  Median(P50): Old = {stats_act_old['p50']:.2f} RUB | New = {stats_act_new['p50']:.2f} RUB")
    print(f"  P90:         Old = {stats_act_old['p90']:.2f} RUB | New = {stats_act_new['p90']:.2f} RUB")
    print(f"  P99:         Old = {stats_act_old['p99']:.2f} RUB | New = {stats_act_new['p99']:.2f} RUB")

    # 4. Correlation & Delta Analysis
    corr_pearson = np.corrcoef(y_old, y_new)[0, 1]
    log_old = np.log1p(y_old)
    log_new = np.log1p(y_new)
    corr_log = np.corrcoef(log_old, log_new)[0, 1]
    deltas = y_new - y_old
    log_deltas = log_new - log_old

    print("\n7. CORRELATION & DELTA AUDIT:")
    print(f"  Pearson Correlation:     {corr_pearson:.6f}")
    print(f"  Log(1+Y) Correlation:    {corr_log:.6f}")
    print(f"  Mean Absolute Delta:     {np.mean(np.abs(deltas)):.4f} RUB")
    print(f"  Median Absolute Delta:   {np.median(np.abs(deltas)):.4f} RUB")
    print(f"  Mean Log Delta:          {np.mean(log_deltas):+.6f}")
    print(f"  Share Increased:         {np.mean(y_new > y_old) * 100:.2f}%")
    print(f"  Share Decreased:         {np.mean(y_new < y_old) * 100:.2f}%")
    print(f"  Share Identical (diff=0):{np.mean(y_new == y_old) * 100:.2f}%")

if __name__ == "__main__":
    main()
