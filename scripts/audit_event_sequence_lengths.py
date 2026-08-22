"""Audit of Event-Day Sequence Lengths and Truncation Span.

Calculates exact distributions of active event-days across 365-day history:
- For Validation (2026-01-14)
- For Train Anchors (representative anchor 2025-12-08 and 2025-09-01)
- For Test (2026-02-13)
- Breakdown by transition states (0->0, 0->1, 1->0, 1->1) on validation
- Evaluates truncation impact at max_events=128
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def audit_anchor_event_counts(
    df_raw_lazy: pl.LazyFrame,
    anchor_str: str,
    user_ids: np.ndarray,
    max_events: int = 128,
) -> pl.DataFrame:
    anchor_dt = pl.lit(anchor_str).str.to_date()
    start_dt = pl.lit(anchor_str).str.to_date().dt.offset_by("-364d")
    user_ids_set = set(user_ids)

    # 1. Filter raw events
    df_filtered = (
        df_raw_lazy
        .filter(
            (pl.col("event_date") >= start_dt)
            & (pl.col("event_date") <= anchor_dt)
            & (pl.col("user_id").is_in(user_ids_set))
        )
        .select(["user_id", "event_date"])
    )

    # 2. Daily unique active days per user
    df_daily = (
        df_filtered.group_by(["user_id", "event_date"])
        .agg(pl.count().alias("count_daily"))
        .sort(["user_id", "event_date"])
    )

    # 3. Aggregate per user: total active days and retained history span
    anchor_dt_py = datetime.strptime(anchor_str, "%Y-%m-%d").date()

    df_user_events = (
        df_daily.group_by("user_id")
        .agg([
            pl.count().alias("n_events_raw"),
            pl.col("event_date").sort().alias("sorted_dates"),
        ])
    ).collect()

    # Build per-user metrics
    user_id_to_stats = {}
    for row in df_user_events.iter_rows(named=True):
        uid = row["user_id"]
        n_raw = row["n_events_raw"]
        dates = row["sorted_dates"]
        
        n_retained = min(n_raw, max_events)
        was_truncated = int(n_raw > max_events)
        
        # Retained history span
        retained_dates = dates[-max_events:] if n_raw > 0 else []
        if len(retained_dates) > 0:
            oldest_retained = retained_dates[0]
            span_days = (anchor_dt_py - oldest_retained).days
        else:
            span_days = 0
            
        user_id_to_stats[uid] = (n_raw, n_retained, was_truncated, span_days)

    rows = []
    for uid in user_ids:
        if uid in user_id_to_stats:
            n_raw, n_ret, was_trunc, span = user_id_to_stats[uid]
        else:
            n_raw, n_ret, was_trunc, span = 0, 0, 0, 0
        rows.append({
            "user_id": uid,
            "anchor": anchor_str,
            "n_events_raw": n_raw,
            "n_events_retained": n_ret,
            "was_truncated": was_trunc,
            "retained_history_span_days": span,
        })

    return pl.DataFrame(rows)


def compute_distribution_stats(df: pl.DataFrame, name: str) -> dict:
    n_raw = df["n_events_raw"].to_numpy()
    span = df["retained_history_span_days"].to_numpy()
    n_total = len(n_raw)

    return {
        "dataset": name,
        "N_samples": n_total,
        "mean_events": float(np.mean(n_raw)),
        "std_events": float(np.std(n_raw)),
        "min_events": int(np.min(n_raw)),
        "P01": float(np.percentile(n_raw, 1)),
        "P10": float(np.percentile(n_raw, 10)),
        "P25": float(np.percentile(n_raw, 25)),
        "P50": float(np.percentile(n_raw, 50)),
        "P75": float(np.percentile(n_raw, 75)),
        "P90": float(np.percentile(n_raw, 90)),
        "P95": float(np.percentile(n_raw, 95)),
        "P99": float(np.percentile(n_raw, 99)),
        "max_events": int(np.max(n_raw)),
        "fraction_zero_events": float(np.mean(n_raw == 0)),
        "fraction_le_16": float(np.mean(n_raw <= 16)),
        "fraction_le_32": float(np.mean(n_raw <= 32)),
        "fraction_le_64": float(np.mean(n_raw <= 64)),
        "fraction_gt_128_truncated": float(np.mean(n_raw > 128)),
        "fraction_gt_192": float(np.mean(n_raw > 192)),
        "fraction_gt_256": float(np.mean(n_raw > 256)),
        "mean_retained_span_days": float(np.mean(span[n_raw > 0])) if (n_raw > 0).sum() > 0 else 0.0,
        "P10_retained_span_days": float(np.percentile(span[n_raw > 0], 10)) if (n_raw > 0).sum() > 0 else 0.0,
        "P50_retained_span_days": float(np.percentile(span[n_raw > 0], 50)) if (n_raw > 0).sum() > 0 else 0.0,
        "P90_retained_span_days": float(np.percentile(span[n_raw > 0], 90)) if (n_raw > 0).sum() > 0 else 0.0,
        "fraction_span_lt_180d_when_truncated": float(np.mean(span[n_raw > 128] < 180)) if (n_raw > 128).sum() > 0 else 0.0,
        "fraction_span_lt_90d_when_truncated": float(np.mean(span[n_raw > 128] < 90)) if (n_raw > 128).sum() > 0 else 0.0,
    }


def main():
    print("[*] Starting Event Sequence Length and Truncation Audit...")
    out_dir = Path("artifacts/ett_optimization")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path("data/train.parquet")
    df_raw_lazy = pl.scan_parquet(train_path)

    users_df = pl.read_parquet("artifacts/selected_users_100k.parquet")
    user_ids = users_df["user_id"].to_numpy()

    # 1. Validation (2026-01-14)
    print("  [*] Auditing Validation Anchor 2026-01-14...")
    df_val = audit_anchor_event_counts(df_raw_lazy, "2026-01-14", user_ids, max_events=128)

    # Load validation transition states
    snap_val = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
    val_was_act = (snap_val["lifetime_gmv"].to_numpy() > 0).astype(int)
    val_buy = (snap_val["target"].to_numpy() > 0).astype(int)
    
    # 0->0, 0->1, 1->0, 1->1
    transitions = np.empty(len(user_ids), dtype=object)
    for i in range(len(user_ids)):
        transitions[i] = f"{val_was_act[i]}->{val_buy[i]}"

    df_val = df_val.with_columns(pl.Series("transition_state", transitions))

    # 2. Train Anchor (2025-12-08)
    print("  [*] Auditing Train Anchor 2025-12-08...")
    df_train_1208 = audit_anchor_event_counts(df_raw_lazy, "2025-12-08", user_ids, max_events=128)

    # 3. Train Anchor (2025-09-01)
    print("  [*] Auditing Train Anchor 2025-09-01...")
    df_train_0901 = audit_anchor_event_counts(df_raw_lazy, "2025-09-01", user_ids, max_events=128)

    # 4. Overall distribution table
    dist_rows = [
        compute_distribution_stats(df_val, "Validation_2026-01-14"),
        compute_distribution_stats(df_train_1208, "Train_Anchor_2025-12-08"),
        compute_distribution_stats(df_train_0901, "Train_Anchor_2025-09-01"),
    ]
    df_dist = pl.DataFrame(dist_rows)
    df_dist.write_csv(out_dir / "event_count_distribution.csv")
    print(f"  [+] Saved {out_dir / 'event_count_distribution.csv'}")

    # 5. Breakdown by transition on validation
    trans_rows = []
    for state in ["0->0", "0->1", "1->0", "1->1"]:
        df_sub = df_val.filter(pl.col("transition_state") == state)
        trans_rows.append(compute_distribution_stats(df_sub, f"Val_Transition_{state}"))

    df_trans = pl.DataFrame(trans_rows)
    df_trans.write_csv(out_dir / "event_count_by_transition.csv")
    print(f"  [+] Saved {out_dir / 'event_count_by_transition.csv'}")

    print("\n" + "=" * 80)
    print("EVENT COUNT AUDIT SUMMARY:")
    print("=" * 80)
    for r in dist_rows:
        print(f"{r['dataset']:25s}: Mean={r['mean_events']:.1f}, P50={r['P50']:.0f}, P90={r['P90']:.0f}, P99={r['P99']:.0f}, >128 Truncated={r['fraction_gt_128_truncated']*100:.2f}%")

    print("\nTRANSITION BREAKDOWN (Validation 2026-01-14):")
    for r in trans_rows:
        print(f"{r['dataset']:25s}: Mean={r['mean_events']:.1f}, P50={r['P50']:.0f}, P90={r['P90']:.0f}, P99={r['P99']:.0f}, >128 Truncated={r['fraction_gt_128_truncated']*100:.2f}%")


if __name__ == "__main__":
    main()
