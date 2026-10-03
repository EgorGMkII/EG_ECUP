"""Per-transition RMSLE analysis for activity-window sweep (F4 fold).

Answers:
  - How many users are 'borderline' (bought 31-90d ago, not 30d)?
  - What fraction of 10/11 groups are borderline?
  - How does RMSLE differ by window for each transition group?
"""
import polars as pl
import numpy as np
from datetime import date, timedelta

raw   = pl.read_parquet("data/train.parquet")
users = pl.read_csv("sample_submit.csv")["user_id"].to_numpy()
anchor = date(2026, 1, 14)

# ── build activity flags ──────────────────────────────────────────────
raw_dates = raw.with_columns(pl.col("event_date").cast(pl.Utf8).str.to_date())

hist = raw_dates.filter(
    pl.col("user_id").is_in(users.tolist()) & (pl.col("event_date") <= anchor)
)
target_raw = raw_dates.filter(
    pl.col("user_id").is_in(users.tolist())
    & (pl.col("event_date") > anchor)
    & (pl.col("event_date") <= date(2026, 2, 13))
)
target_gmv = target_raw.group_by("user_id").agg(pl.col("gmv").sum().alias("target_gmv"))

base = pl.DataFrame({"user_id": users})

for w in [90, 60, 30]:
    wstart = anchor - timedelta(days=w - 1)
    win = hist.filter(pl.col("event_date") >= wstart)
    grp = win.group_by("user_id").agg(pl.col("gmv").sum().alias(f"gmv_{w}d"))
    base = base.join(grp, on="user_id", how="left").with_columns(
        pl.col(f"gmv_{w}d").fill_null(0.0)
    )

base = base.join(target_gmv, on="user_id", how="left").with_columns(
    pl.col("target_gmv").fill_null(0.0)
)
base = base.with_columns(
    [
        (pl.col("gmv_90d") > 0).alias("act90"),
        (pl.col("gmv_60d") > 0).alias("act60"),
        (pl.col("gmv_30d") > 0).alias("act30"),
        (pl.col("target_gmv") > 0).alias("will_buy"),
    ]
)
# borderline: bought 31-90d ago but NOT in last 30d
base = base.with_columns(
    (pl.col("act90") & ~pl.col("act30")).alias("borderline"),
    (pl.col("act90") & pl.col("act30")).alias("recent_30d"),
)

act90     = base["act90"].to_numpy()
act60     = base["act60"].to_numpy()
act30     = base["act30"].to_numpy()
wb        = base["will_buy"].to_numpy()
border    = base["borderline"].to_numpy()
n         = len(base)

# ── Borderline segment profile ────────────────────────────────────────
print("=" * 70)
print("BORDERLINE SEGMENT  (bought 31-90d ago, not in last 30d)")
print("=" * 70)
bn = int(border.sum())
bwb = int((border & wb).sum())
print(f"  N = {bn:,}  ({bn/n:.1%} of 250k)")
print(f"  Will buy in next 30d: {bwb:,}  ({bwb/bn:.1%})")
print()

bl_df = base.filter(pl.col("borderline"))
bl_buy_df = base.filter(pl.col("borderline") & pl.col("will_buy"))
print(f"  GMV in 90d window  — median: {bl_df['gmv_90d'].median():.0f}  mean: {bl_df['gmv_90d'].mean():.0f}")
print(f"  Future GMV (buyers) — median: {bl_buy_df['target_gmv'].median():.0f}  mean: {bl_buy_df['target_gmv'].mean():.0f}")
print()

for label, mask in [("10 (active->churn)", act90 & ~wb), ("11 (retained)", act90 & wb)]:
    bl_in = int((mask & border).sum())
    tot = int(mask.sum())
    print(f"  {label}: {bl_in:,}/{tot:,} are borderline = {bl_in/tot:.1%}")

print()
print("Window effect on active cohort size:")
for w, a_mask in [(90, act90), (60, act60), (30, act30)]:
    a = int(a_mask.sum())
    lost_11 = int((act90 & wb & ~a_mask).sum())  # retained buyers reclassified to inactive
    lost_10 = int((act90 & ~wb & ~a_mask).sum())  # churners reclassified to inactive
    print(f"  w{w}: active={a:,} ({a/n:.1%}) | 11-group lost to inactive={lost_11:,} | 10-group lost to inactive={lost_10:,}")

# ── Per-transition RMSLE ──────────────────────────────────────────────
print()
print("=" * 70)
print("PER-TRANSITION RMSLE  (F4 fold, w90 transition definition)")
print("=" * 70)

models = {
    "CB_baseline": "artifacts/direct_temporal_cv_v1/experiments/direct_cv_catboost_baseline_v1/fold_F4_predictions.parquet",
    "w90_specialist": "artifacts/direct_temporal_cv_v1/experiments/direct_cv_catboost_cohort_specialist_v1/fold_F4_predictions.parquet",
    "w60_specialist": "artifacts/direct_temporal_cv_v1/experiments/direct_cv_catboost_cohort_specialist_w60_v1/fold_F4_predictions.parquet",
    "w30_specialist": "artifacts/direct_temporal_cv_v1/experiments/direct_cv_catboost_cohort_specialist_w30_v1/fold_F4_predictions.parquet",
}

masks = {
    "00":     ~act90 & ~wb,
    "01":     ~act90 & wb,
    "10":      act90 & ~wb,
    "11":      act90 & wb,
    "10_borderline":  act90 & ~wb & border,
    "10_recent":      act90 & ~wb & ~border,
    "11_borderline":  act90 & wb  & border,
    "11_recent":      act90 & wb  & ~border,
}

y_true_z = np.log1p(base["target_gmv"].to_numpy())

cols = list(masks.keys()) + ["Total"]
header = f"{'Model':<16}" + "".join(f"  {c:>13}" for c in cols)
print(header)
print("-" * len(header))

for mname, fpath in models.items():
    preds = pl.read_parquet(fpath)
    merged = base.select(["user_id"]).join(
        preds.select(["user_id", "prediction_z"]), on="user_id", how="left"
    )
    pred_z = merged["prediction_z"].to_numpy()
    sq_err = (pred_z - y_true_z) ** 2
    vals = []
    for key, m in masks.items():
        vals.append(f"{np.sqrt(sq_err[m].mean()):.4f}" if m.sum() > 0 else "  nan")
    vals.append(f"{np.sqrt(sq_err.mean()):.6f}")
    print(f"{mname:<16}" + "".join(f"  {v:>13}" for v in vals))

print()
# Also show: prediction bias for 10 and 11 groups (mean prediction vs mean target)
print("=" * 70)
print("PREDICTION BIAS by group (mean prediction_z vs mean target_z)")
print("=" * 70)
for mname, fpath in models.items():
    preds = pl.read_parquet(fpath)
    merged = base.select(["user_id"]).join(
        preds.select(["user_id", "prediction_z"]), on="user_id", how="left"
    )
    pred_z = merged["prediction_z"].to_numpy()
    print(f"\n{mname}:")
    for label, m in [("10", act90 & ~wb), ("11", act90 & wb), ("11_borderline", act90 & wb & border)]:
        bias = float(np.mean(pred_z[m]) - np.mean(y_true_z[m]))
        print(f"  {label:<16}: mean_pred_z={np.mean(pred_z[m]):.4f}  mean_true_z={np.mean(y_true_z[m]):.4f}  bias={bias:+.4f}")
