"""Script 03: Build & Validate Expanding-Window Temporal Folds (Time-CV)."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import polars as pl
from src.specialized_hurdle.definitions import ALL_AVAILABLE_ANCHORS
from src.specialized_hurdle.temporal_splits import build_meta_oof_folds


def main():
    print("=" * 80)
    print("03: BUILD & VALIDATE EXPANDING-WINDOW TEMPORAL FOLDS")
    print("=" * 80)

    folds_dir = Path("artifacts/specialized_hurdle/folds")
    folds_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    folds = build_meta_oof_folds(all_anchors=ALL_AVAILABLE_ANCHORS)
    fold_summary_rows = []

    print(f"[*] Generated {len(folds)} Expanding-Window Meta-OOF Folds:\n")

    for fold in folds:
        fold_subdir = folds_dir / fold.fold_id
        fold_subdir.mkdir(parents=True, exist_ok=True)

        outer_dt = datetime.strptime(fold.outer_anchor, "%Y-%m-%d").date()
        val_target_start = outer_dt + timedelta(days=1)
        val_target_end = outer_dt + timedelta(days=30)

        # Strict leakage check
        max_train_target_end = None
        for a in fold.train_anchors:
            a_dt = datetime.strptime(a, "%Y-%m-%d").date()
            t_end = a_dt + timedelta(days=30)
            if max_train_target_end is None or t_end > max_train_target_end:
                max_train_target_end = t_end
            assert t_end <= outer_dt, f"Leakage detected in {fold.fold_id}: anchor {a} target ends {t_end} > outer {outer_dt}"

        print(f"[{fold.fold_id}] Outer Anchor: {fold.outer_anchor} (Val Target: {val_target_start} .. {val_target_end})")
        print(f"   Train Anchors ({fold.n_train_anchors}): {fold.train_anchors[0]} .. {fold.train_anchors[-1]}")
        print(f"   Max Train Target End: {max_train_target_end} (Gap to Val Target: {(val_target_start - max_train_target_end).days} days)")
        print(f"   Inner Val Anchor: {fold.inner_val_anchor} (Inner Train: {len(fold.inner_train_anchors)} anchors)")
        print()

        fold_config = {
            "fold_id": fold.fold_id,
            "outer_anchor": fold.outer_anchor,
            "train_anchors": fold.train_anchors,
            "n_train_anchors": fold.n_train_anchors,
            "inner_val_anchor": fold.inner_val_anchor,
            "inner_train_anchors": fold.inner_train_anchors,
            "val_target_start": str(val_target_start),
            "val_target_end": str(val_target_end),
            "max_train_target_end": str(max_train_target_end),
        }

        with open(fold_subdir / "fold_config.json", "w", encoding="utf-8") as f:
            json.dump(fold_config, f, indent=2)

        fold_summary_rows.append({
            "fold_id": fold.fold_id,
            "outer_anchor": fold.outer_anchor,
            "n_train_anchors": fold.n_train_anchors,
            "first_train_anchor": fold.train_anchors[0],
            "last_train_anchor": fold.train_anchors[-1],
            "max_train_target_end": str(max_train_target_end),
            "val_target_start": str(val_target_start),
            "val_target_end": str(val_target_end),
            "zero_leakage_verified": True,
        })

    df_folds = pl.DataFrame(fold_summary_rows)
    out_csv = reports_dir / "temporal_folds_summary.csv"
    df_folds.write_csv(out_csv)
    print(f"[+] Saved temporal folds summary to {out_csv}")


if __name__ == "__main__":
    main()
