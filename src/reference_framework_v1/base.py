"""Public adapter and recipe types for reference experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch

from src.ssl_temporal_stack_v1.stores import StoreRegistry


@dataclass(frozen=True)
class PredictionSpec:
    model_id: str
    react_column: str | None
    churn_column: str | None
    amount_column: str | None
    direct_column: str | None = None


@dataclass(frozen=True)
class RunContext:
    run_name: str
    train_anchors: tuple[str, ...]
    holdout_anchor: str
    users: tuple[int, ...]
    stores: StoreRegistry
    device: torch.device
    root_seed: int
    output_dir: Path
    anchor_tickets: dict[str, int] | None = None
    raw_events: pl.DataFrame | None = None


@dataclass
class ModelResult:
    model_id: str
    predictions: dict[str, np.ndarray]
    training_report: dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    values: Mapping[str, Any]


class FirstLevelAdapter(ABC):
    """Independent first-level model used by both RUN A and RUN B."""

    model_id: str
    required_stores: frozenset[str]
    prediction_spec: PredictionSpec

    @abstractmethod
    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        raise NotImplementedError
