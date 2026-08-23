"""Dynamic prediction-bank schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl

from .base import PredictionSpec


@dataclass(frozen=True)
class PredictionSchema:
    react_columns: tuple[str, ...]
    churn_columns: tuple[str, ...]
    amount_columns: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.react_columns + self.churn_columns + self.amount_columns


def schema_from_specs(specs: list[PredictionSpec]) -> PredictionSchema:
    schema = PredictionSchema(
        react_columns=tuple(spec.react_column for spec in specs if spec.react_column),
        churn_columns=tuple(spec.churn_column for spec in specs if spec.churn_column),
        amount_columns=tuple(spec.amount_column for spec in specs if spec.amount_column),
    )
    if not schema.react_columns or not schema.churn_columns or not schema.amount_columns:
        raise ValueError("At least one React, Churn, and Amount prediction is required")
    if len(set(schema.all_columns)) != len(schema.all_columns):
        raise ValueError("Prediction columns must be unique")
    return schema


def validate_prediction_mapping(predictions: Mapping[str, np.ndarray], schema: PredictionSchema, rows: int) -> None:
    expected, observed = set(schema.all_columns), set(predictions)
    if expected != observed:
        raise ValueError(f"Prediction schema mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    for name in schema.all_columns:
        values = np.asarray(predictions[name])
        if values.shape != (rows,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid prediction column: {name}")


def make_prediction_bank(frame: pl.DataFrame, anchor: str, predictions: Mapping[str, np.ndarray], schema: PredictionSchema) -> pl.DataFrame:
    required = {"user_id", "was_active", "will_buy", "future_gmv_30d", "z_target"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Holdout frame missing {sorted(missing)}")
    if frame["user_id"].n_unique() != frame.height:
        raise ValueError("Holdout user IDs are not unique")
    validate_prediction_mapping(predictions, schema, frame.height)
    bank = frame.select(["user_id", "was_active", "will_buy", "future_gmv_30d", "z_target"]).with_columns(pl.lit(anchor).alias("anchor"))
    return bank.with_columns([pl.Series(name, np.asarray(predictions[name], dtype=np.float64)) for name in schema.all_columns]).select([
        "user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "z_target", *schema.all_columns,
    ])


def bank_arrays(bank: pl.DataFrame, schema: PredictionSchema) -> dict[str, np.ndarray]:
    return {
        "react": bank.select(schema.react_columns).to_numpy().astype(np.float64, copy=False),
        "churn": bank.select(schema.churn_columns).to_numpy().astype(np.float64, copy=False),
        "amount": bank.select(schema.amount_columns).to_numpy().astype(np.float64, copy=False),
        "active": bank["was_active"].to_numpy().astype(np.int8, copy=False),
        "target": bank["z_target"].to_numpy().astype(np.float64, copy=False),
    }
