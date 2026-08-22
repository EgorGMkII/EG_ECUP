import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.sequential.gru_sweep import get_anchor_set

OUT_DIR = Path("artifacts/user_embedding")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load train.parquet and extract all unique user_ids
print("[*] Loading train.parquet to extract 250,000 unique user_ids...")
train_df = pl.read_parquet("data/train.parquet")
unique_user_ids = sorted(train_df["user_id"].unique().to_list())
n_users = len(unique_user_ids)
assert n_users == 250000, f"Expected 250,000 users, got {n_users}"

# 1-based index (0 = UNK)
user_indices = np.arange(1, n_users + 1, dtype=np.int64)

mapping_df = pl.DataFrame({
    "user_id": unique_user_ids,
    "user_idx": user_indices,
})
mapping_df.write_parquet(OUT_DIR / "user_id_mapping.parquet")
print(f"[+] Saved user_id_mapping.parquet ({n_users} users, user_idx in [{user_indices.min()}, {user_indices.max()}])")

# Compute checksum
user_ids_bytes = np.array(unique_user_ids, dtype=np.int64).tobytes()
checksum = hashlib.sha256(user_ids_bytes).hexdigest()

meta = {
    "num_users": n_users,
    "min_user_idx": int(user_indices.min()),
    "max_user_idx": int(user_indices.max()),
    "unk_idx": 0,
    "num_embeddings": n_users + 1,
    "checksum_sha256": checksum,
    "created_at": datetime.now().isoformat(),
    "source_file": "data/train.parquet",
}
with open(OUT_DIR / "user_id_mapping_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"[+] Saved user_id_mapping_meta.json (SHA256: {checksum[:16]}...)")

# 2. Canonical GRU-180 Config
anchors_14 = [str(a) for a in get_anchor_set("recent_14")]
train_anchors = anchors_14[:-1]
val_anchor = anchors_14[-1]

canonical_config = {
    "model_type": "GRU-180",
    "sequence_length": 180,
    "input_dim": 15,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.15,
    "use_attention": True,
    "anchor_set": "recent_14",
    "train_anchors": train_anchors,
    "validation_anchor": val_anchor,
    "selected_user_file": "artifacts/selected_users_100k.parquet",
    "user_id_mapping_path": "artifacts/user_embedding/user_id_mapping.parquet",
    "channel_order": [
        "searches", "cart_adds", "orders", "gmv", "discount_amt",
        "avg_item_price", "views_per_search", "cart_to_order_ratio", "orders_per_search",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend", "is_holiday"
    ],
    "heads": ["reactivation", "churn", "buy", "conditional_reg", "direct_reg"],
    "hurdle_formula": "p_buy = where(past_b == 0, p_react, 1 - p_churn); z_fact = (p_buy ** alpha) * z_cond; pred_rub = expm1(z_fact)",
    "lambda_cls": 0.50,
    "lambda_reg": 1.00,
    "alpha": 1.10,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "epochs": 10,
    "batch_size": 2048,
    "clip_grad_norm": 1.0,
    "random_seeds": [42, 43, 44],
    "scaler": "SequentialScaler (Channel-wise mean and std computed on training anchors)",
    "prediction_space": "z = log1p(GMV), rub = expm1(z)",
    "DataSphere_environment": "Python 3.10.13, PyTorch 2.1+ CUDA, A100 GPU",
}
with open(OUT_DIR / "canonical_gru180_config.json", "w") as f:
    json.dump(canonical_config, f, indent=2)
print("[+] Saved canonical_gru180_config.json")
