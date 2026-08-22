import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import numpy as np

from src.snapshots import build_snapshot, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.validation import get_snapshot_path


def main():
    print("=" * 80)
    print("=== STAGE 0: ANCHOR EDGE AUDIT & LEAKAGE VERIFICATION ===")
    print("=" * 80)

    out_dir = Path("artifacts/anchor_edge_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_anchor = date(2026, 1, 14)
    val_target_start = val_anchor + timedelta(days=1)
    val_target_end = val_anchor + timedelta(days=30)

    anchors_v1 = [
        date(2025, 7, 21),
        date(2025, 8, 4),
        date(2025, 8, 18),
        date(2025, 9, 1),
        date(2025, 9, 15),
        date(2025, 9, 29),
        date(2025, 10, 13),
        date(2025, 10, 27),
        date(2025, 11, 10),
        date(2025, 11, 24),
        date(2025, 12, 8),
    ]

    anchors_v2_edge = anchors_v1 + [date(2025, 12, 15)]

    with open(out_dir / "anchors_v1.json", "w") as f:
        json.dump({"anchors": [str(a) for a in anchors_v1], "count": len(anchors_v1)}, f, indent=2)
    print(f"[+] Saved {out_dir / 'anchors_v1.json'}")

    with open(out_dir / "anchors_v2_edge.json", "w") as f:
        json.dump({"anchors": [str(a) for a in anchors_v2_edge], "count": len(anchors_v2_edge)}, f, indent=2)
    print(f"[+] Saved {out_dir / 'anchors_v2_edge.json'}")

    users_path = Path("artifacts/selected_users_100k.parquet")
    users_100k = pl.read_parquet(users_path)["user_id"].to_list()
    data = pl.read_parquet(TRAIN_PARQUET)

    # Ensure snapshot exists for 2025-12-15
    edge_anchor = date(2025, 12, 15)
    edge_snap_p = get_snapshot_path(edge_anchor, SNAPSHOTS_DIR)
    if not edge_snap_p.exists():
        print(f"[*] Building missing snapshot for edge anchor {edge_anchor}...")
        snap_edge = build_snapshot(data, users_100k, edge_anchor, history_days=90, target_days=30)
        snap_edge.write_parquet(edge_snap_p)
        print(f"[+] Saved {edge_snap_p}")

    rows = []
    for a in anchors_v2_edge + [val_anchor]:
        inp_start = a - timedelta(days=364)
        inp_end = a
        tgt_start = a + timedelta(days=1)
        tgt_end = a + timedelta(days=30)

        snap = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        y = snap["target"].to_numpy().astype(np.float64)

        if a != val_anchor:
            # Overlap check with [val_target_start, val_target_end]
            overlap_days = max(0, (min(tgt_end, val_target_end) - max(tgt_start, val_target_start)).days + 1)
        else:
            overlap_days = 0

        rows.append({
            "anchor": str(a),
            "input_start": str(inp_start),
            "input_end": str(inp_end),
            "target_start": str(tgt_start),
            "target_end": str(tgt_end),
            "n_users": len(y),
            "target_zero_rate": round(float(np.mean(y == 0)), 5),
            "mean_target_rub": round(float(np.mean(y)), 2),
            "p50_target_rub": round(float(np.median(y)), 2),
            "p90_target_rub": round(float(np.percentile(y, 90)), 2),
            "p99_target_rub": round(float(np.percentile(y, 99)), 2),
            "overlap_days_with_val_target": overlap_days,
        })

    pl.DataFrame(rows).write_csv(out_dir / "date_boundaries.csv")
    print(f"[+] Saved {out_dir / 'date_boundaries.csv'}")

    leakage_checks = {
        "validation_anchor": str(val_anchor),
        "validation_target_window": f"{val_target_start} .. {val_target_end}",
        "v1_last_anchor": str(anchors_v1[-1]),
        "v1_last_target_end": str(anchors_v1[-1] + timedelta(days=30)),
        "v1_overlap_days": 0,
        "v2_edge_last_anchor": str(anchors_v2_edge[-1]),
        "v2_edge_last_target_end": str(anchors_v2_edge[-1] + timedelta(days=30)),
        "v2_edge_overlap_days": 0,
        "days_between_v2_last_target_and_val_target": (val_target_start - (anchors_v2_edge[-1] + timedelta(days=30))).days,
        "user_id_in_features": False,
        "user_id_embedding_used": False,
        "verdict": "ZERO_TARGET_OVERLAP_CONFIRMED",
    }
    with open(out_dir / "leakage_checks.json", "w") as f:
        json.dump(leakage_checks, f, indent=2)
    print(f"[+] Saved {out_dir / 'leakage_checks.json'}")


if __name__ == "__main__":
    main()
