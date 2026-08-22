"""Restore data/submission.csv to Rock-Solid Submission v5.1 (LB: 1.68028888)."""

import polars as pl
from pathlib import Path

def main():
    test_parquet_path = Path("artifacts/transformer_audit/v51_exact_test.parquet")
    df_v51 = pl.read_parquet(test_parquet_path)
    
    sub_df = pl.DataFrame({
        "user_id": df_v51["user_id"],
        "predict": df_v51["predict"],
    })
    
    sub_df.write_csv("data/submission.csv")
    print("[+] Successfully restored data/submission.csv to Submission v5.1 (LB: 1.68028888)!")
    print(f"    - Count: {sub_df.height:,}")
    print(f"    - P50:   {float(sub_df['predict'].median()):.2f} руб.")
    print(f"    - Mean:  {float(sub_df['predict'].mean()):.2f} руб.")
    print(f"    - P90:   {float(sub_df['predict'].quantile(0.90)):.2f} руб.")
    print(f"    - P99:   {float(sub_df['predict'].quantile(0.99)):.2f} руб.")
    print(f"    - Max:   {float(sub_df['predict'].max()):.2f} руб.")

if __name__ == "__main__":
    main()
