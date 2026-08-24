"""Public types and direct-model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import TemporalFold


@dataclass(frozen=True)
class ModelRequirements:
    tabular_features: bool = False
    daily_tensor: bool = False
    event_sequences: bool = False
    btyd_features: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    values: Mapping[str, Any]


@dataclass
class FoldContext:
    fold: TemporalFold
    users: np.ndarray
    train_target_z: np.ndarray
    validation_target_z: np.ndarray
    train_tabular: Any | None
    validation_tabular: Any | None
    train_daily: Any | None
    validation_daily: Any | None
    train_events: Any | None
    validation_events: Any | None
    device: torch.device
    output_dir: Path
    root_seed: int


@dataclass
class FoldPrediction:
    model_id: str
    user_ids: np.ndarray
    prediction_z: np.ndarray
    training_report: dict[str, Any]


class DirectModelAdapter(ABC):
    """A direct log1p(GMV30) model, independently fit per temporal fold."""

    model_id: str
    requirements: ModelRequirements

    @abstractmethod
    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        raise NotImplementedError

    @abstractmethod
    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        raise NotImplementedError
