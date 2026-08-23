"""Immutable first-level prediction schema and bank assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl


MODEL_ORDER = ("cb", "s1", "s2", "ett")
REACT_COLUMNS = tuple(f"{model}_react_logit" for model in MODEL_ORDER)
CHURN_COLUMNS = tuple(f"{model}_churn_logit" for model in MODEL_ORDER)
AMOUNT_COLUMNS = tuple(f"{model}_amount_z" for model in MODEL_ORDER)
PREDICTION_COLUMNS = REACT_COLUMNS + CHURN_COLUMNS + AMOUNT_COLUMNS


@dataclass(frozen=True)
class PredictionSchema:
    react_columns: tuple[str, ...] = REACT_COLUMNS
    churn_columns: tuple[str, ...] = CHURN_COLUMNS
    amount_columns: tuple[str, ...] = AMOUNT_COLUMNS

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.react_columns + self.churn_columns + self.amount_columns


SCHEMA = PredictionSchema()


def validate_prediction_mapping(
    predictions: Mapping[str, np.ndarray],
    *,
    expected_rows: int,
    schema: PredictionSchema = SCHEMA,
) -> None:
    observed = set(predictions)
    expected = set(schema.all_columns)
    if observed != expected:
        raise ValueError(
            f"Prediction columns differ: missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    for name in schema.all_columns:
        values = np.asarray(predictions[name])
        if values.shape != (expected_rows,):
            raise ValueError(f"{name} shape is {values.shape}, expected {(expected_rows,)}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")


def make_prediction_bank(
    holdout: pl.DataFrame,
    anchor: str,
    predictions: Mapping[str, np.ndarray],
    *,
    schema: PredictionSchema = SCHEMA,
) -> pl.DataFrame:
    required = {
        "user_id", "was_active", "will_buy", "future_gmv_30d", "z_target",
    }
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"Holdout frame lacks bank columns: {sorted(missing)}")
    if holdout["user_id"].n_unique() != holdout.height:
        raise ValueError("Holdout frame contains duplicate users")
    validate_prediction_mapping(predictions, expected_rows=holdout.height, schema=schema)
    bank = holdout.select([
        "user_id", "was_active", "will_buy", "future_gmv_30d", "z_target",
    ]).with_columns(pl.lit(anchor).alias("anchor"))
    bank = bank.with_columns([
        pl.Series(name, np.asarray(predictions[name], dtype=np.float64))
        for name in schema.all_columns
    ])
    return bank.select([
        "user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "z_target",
        *schema.all_columns,
    ])


def bank_arrays(bank: pl.DataFrame, schema: PredictionSchema = SCHEMA) -> dict[str, np.ndarray]:
    missing = (set(schema.all_columns) | {"was_active", "z_target"}) - set(bank.columns)
    if missing:
        raise ValueError(f"Prediction bank is missing: {sorted(missing)}")
    return {
        "react": bank.select(schema.react_columns).to_numpy().astype(np.float64, copy=False),
        "churn": bank.select(schema.churn_columns).to_numpy().astype(np.float64, copy=False),
        "amount": bank.select(schema.amount_columns).to_numpy().astype(np.float64, copy=False),
        "active": bank["was_active"].to_numpy().astype(np.int8, copy=False),
        "target": bank["z_target"].to_numpy().astype(np.float64, copy=False),
    }
