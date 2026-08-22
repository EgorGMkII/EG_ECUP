import json
import polars as pl
from pathlib import Path

audit_dir = Path("artifacts/catboost_cadence_audit")

summary_df = pl.read_csv(audit_dir / "catboost_cadence_btyd_summary.csv")
print("=== SUMMARY METRICS ===")
print(summary_df)

b0_metrics = json.loads((audit_dir / "B0_standalone_btyd_metrics.json").read_text())
print("\n=== B0 METRICS ===")
print(json.dumps(b0_metrics, indent=2))

group_ablation = pl.read_csv(audit_dir / "baseline_group_ablation.csv")
print("\n=== BASELINE GROUP ABLATION ===")
print(group_ablation)

checklist = pl.read_csv(audit_dir / "signal_audit_checklist.csv")
print("\n=== CHECKLIST STATUS COUNTS ===")
print(checklist["status"].value_counts())
