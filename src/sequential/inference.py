"""Inference and Test Snapshot Prediction Pipeline for Sequential Models."""

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.sequential.dataset import get_cached_sequence_tensor
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import TRAIN_PARQUET


def predict_test_snapshot(
    model: nn.Module,
    scaler: SequentialScaler,
    test_user_ids: List[int],
    data: Optional[pl.DataFrame] = None,
    anchor_date: date = date(2026, 2, 13),
    seq_len: int = 90,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> pl.DataFrame:
    """Computes final 30d GMV predictions for test users using trained sequential model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data is None:
        data = pl.read_parquet(TRAIN_PARQUET)

    print(f"[*] Extracting test sequence tensor for {len(test_user_ids):,} users...")
    test_tensor = get_cached_sequence_tensor(data, test_user_ids, anchor_date, seq_len=seq_len)
    test_tensor_scaled = scaler.transform(test_tensor)

    model = model.to(device)
    model.eval()

    preds_list = []
    n_samples = len(test_user_ids)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = torch.from_numpy(test_tensor_scaled[i : i + batch_size]).float().to(device)
            out = model(batch_x)

            if len(out) == 2:  # Direct GRU: (z_pred, emb)
                z_p = torch.clamp(out[0], min=0.0).cpu().numpy()
            else:  # Multi-task GRU: (p_logits, z_cond, emb)
                p_buy = torch.sigmoid(out[0]).cpu().numpy()
                z_c = torch.clamp(out[1], min=0.0).cpu().numpy()
                z_p = np.power(p_buy, 1.1) * z_c

            preds_list.append(np.expm1(z_p))

    final_predictions = np.concatenate(preds_list)

    return pl.DataFrame({
        "user_id": test_user_ids,
        "predict_gru": final_predictions,
    })
