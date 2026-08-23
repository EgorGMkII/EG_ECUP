"""Reserved tabular Residual MLP architecture and leakage-safe scaler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResidualMLPRecipe:
    input_features: int = 374
    hidden: int = 512
    blocks: int = 4
    dropout: float = 0.10


class StreamingFeatureScaler:
    """Fit only on a run's train-anchor rows; never reuse RUN A state in RUN B."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def partial_fit(self, values: np.ndarray) -> "StreamingFeatureScaler":
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Scaler requires a rank-2 feature matrix")
        finite = np.where(np.isfinite(data), data, 0.0)
        batch_count = finite.shape[0]
        batch_mean = finite.mean(axis=0)
        batch_m2 = np.square(finite - batch_mean).sum(axis=0)
        if self.mean is None:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return self
        assert self.m2 is not None
        delta, total = batch_mean - self.mean, self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise RuntimeError("Scaler is not fitted")
        scale = np.sqrt(np.maximum(self.m2 / self.count, 1e-12))
        result = (np.asarray(values, dtype=np.float32) - self.mean.astype(np.float32)) / scale.astype(np.float32)
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: Tensor) -> Tensor:
        return self.norm(values + self.layers(values))


class ResidualMLPTransitionBase(nn.Module):
    implementation_id = "residual_mlp_tabular374_v1"

    def __init__(self, recipe: ResidualMLPRecipe = ResidualMLPRecipe()) -> None:
        super().__init__()
        self.recipe = recipe
        self.input = nn.Sequential(nn.Linear(recipe.input_features, recipe.hidden), nn.LayerNorm(recipe.hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualMLPBlock(recipe.hidden, recipe.dropout) for _ in range(recipe.blocks)])
        self.embedding = nn.Linear(recipe.hidden, 128)
        self.reactivation = nn.Linear(128, 1)
        self.churn = nn.Linear(128, 1)
        self.amount = nn.Linear(128, 1)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.recipe.input_features:
            raise ValueError("Expected [batch, 374] tabular features")
        representation = nn.functional.gelu(self.embedding(self.blocks(self.input(features))))
        return {"reactivation_logit": self.reactivation(representation).squeeze(-1), "churn_logit": self.churn(representation).squeeze(-1), "amount_z": self.amount(representation).squeeze(-1)}
