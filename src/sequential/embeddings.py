"""User-level Sequential Embedding Extraction and GBDT Stacking."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
import torch.nn as nn

EMB_DIR = Path("artifacts/sequential_embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)


def extract_embeddings_dataframe(
    model: nn.Module,
    tensor: np.ndarray,
    user_ids: List[int],
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> pl.DataFrame:
    """Extracts hidden representations [N, H] from sequential encoder as Polars DataFrame."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    embs_list = []
    n_samples = len(tensor)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = torch.from_numpy(tensor[i : i + batch_size]).float().to(device)
            # Forward call returns emb as second or third element
            out = model(batch_x)
            emb = out[-1]  # [B, H]
            embs_list.append(emb.cpu().numpy())

    all_embs = np.vstack(embs_list).astype(np.float32)
    h_dim = all_embs.shape[1]

    data_dict = {"user_id": user_ids}
    for col_idx in range(h_dim):
        data_dict[f"seq_emb_{col_idx:03d}"] = all_embs[:, col_idx]

    return pl.DataFrame(data_dict)
