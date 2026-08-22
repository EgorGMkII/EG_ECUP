"""Preprocessing, scaling, and transformation tools for 3D Sequential Datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

CHANNELS = [
    "searches",
    "to_cart",
    "to_ord",
    "gmv",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
    "is_active",
    "is_purchase_day",
    "sin_dow",
    "cos_dow",
    "normalized_position",
]

NUMERIC_CHANNELS = [
    "searches",
    "to_cart",
    "to_ord",
    "gmv",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
]


class SequentialScaler:
    """Channel-wise robust scaler fit strictly on training anchor sequences."""

    def __init__(self, channels: List[str] = CHANNELS):
        self.channels = channels
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fit: bool = False

    def fit(self, tensor: np.ndarray) -> "SequentialScaler":
        """Fits channel-wise mean and std on 3D tensor [N_samples, T_sequence, N_channels]."""
        # tensor shape: [N, T, C]
        flat_tensor = tensor.reshape(-1, tensor.shape[-1])
        # Compute mean and std over active elements or full slice
        self.mean = np.mean(flat_tensor, axis=0).astype(np.float32)
        self.std = np.std(flat_tensor, axis=0).astype(np.float32)
        # Avoid division by zero
        self.std[self.std < 1e-5] = 1.0

        # Leave binary and periodic channels unscaled (zero mean, unit scale)
        for i, ch in enumerate(self.channels):
            if ch in ["is_active", "is_purchase_day", "sin_dow", "cos_dow", "normalized_position"]:
                self.mean[i] = 0.0
                self.std[i] = 1.0

        self.is_fit = True
        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """Transforms 3D tensor in-place or returns transformed copy."""
        assert self.is_fit, "SequentialScaler must be fitted before calling transform."
        return ((tensor - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, tensor: np.ndarray) -> np.ndarray:
        return self.fit(tensor).transform(tensor)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves scaler state to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "channels": self.channels,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "is_fit": self.is_fit,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SequentialScaler":
        """Loads scaler state from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(channels=data["channels"])
        scaler.mean = np.array(data["mean"], dtype=np.float32) if data["mean"] else None
        scaler.std = np.array(data["std"], dtype=np.float32) if data["std"] else None
        scaler.is_fit = data["is_fit"]
        return scaler
