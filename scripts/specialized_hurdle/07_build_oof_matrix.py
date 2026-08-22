"""Script 07: Assemble Full OOF Matrix from Fold Predictions."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import polars as pl


def main():
    print("=" * 80)
    print("07: ASSEMBLE FULL WALK-FORWARD OOF MATRIX")
    print("=" * 80)

    oof_dir = Path("artifacts/specialized_hurdle/oof")
    fold_files = sorted(list(oof_dir.glob("fold_*.parquet")))

    print(f"[*] Found {len(fold_files)} fold prediction files: {[f.name for f in fold_files]}")
    oof_chunks = []
    for f in fold_files:
        df = pl.read_parquet(f)
        oof_chunks.append(df)

    df_full_oof = pl.concat(oof_chunks)
    out_path = oof_dir / "full_walkforward_oof_matrix.parquet"
    df_full_oof.write_parquet(out_path)

    print(f"\n[+] Saved full walk-forward OOF matrix ({len(df_full_oof):,} rows) to {out_path}")


if __name__ == "__main__":
    main()
