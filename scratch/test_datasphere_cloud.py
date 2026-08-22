import os
import torch
import polars as pl
from pathlib import Path

print("========================================")
print("=== DATASPHERE CLOUD SANITY CHECK ===")
print("========================================")
print(f"[*] Python version: {os.sys.version}")
print(f"[*] PyTorch version: {torch.__version__}")
print(f"[*] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[*] GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"[*] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

train_path = Path("data/train.parquet")
if train_path.exists():
    df = pl.scan_parquet(train_path).select(pl.len()).collect()
    print(f"[+] Successfully loaded train.parquet: {df['len'][0]:,} rows!")
else:
    print("[-] train.parquet not found in job snapshot!")

out_dir = Path("artifacts/test_datasphere")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "sanity_check.txt", "w") as f:
    f.write("DataSphere Sanity Check Passed Successfully!\n")

print("[+] DataSphere Sanity Check Completed!")
