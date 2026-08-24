"""Immutable four-fold temporal protocol for the 250k template cohort."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TemporalFold:
    """One train-snapshot -> future-snapshot evaluation fold."""

    fold_id: str
    inference_anchor: date

    @property
    def train_anchor(self) -> date:
        return self.inference_anchor - timedelta(days=30)

    @property
    def train_target_start(self) -> date:
        return self.train_anchor + timedelta(days=1)

    @property
    def train_target_end(self) -> date:
        return self.inference_anchor

    @property
    def validation_target_start(self) -> date:
        return self.inference_anchor + timedelta(days=1)

    @property
    def validation_target_end(self) -> date:
        return self.inference_anchor + timedelta(days=30)

    def as_dict(self) -> dict[str, str]:
        return {
            "fold_id": self.fold_id,
            "train_anchor": self.train_anchor.isoformat(),
            "train_target_start": self.train_target_start.isoformat(),
            "train_target_end": self.train_target_end.isoformat(),
            "inference_anchor": self.inference_anchor.isoformat(),
            "validation_target_start": self.validation_target_start.isoformat(),
            "validation_target_end": self.validation_target_end.isoformat(),
        }


FOUR_FOLD_250K_V1: tuple[TemporalFold, ...] = (
    TemporalFold("F1", date(2025, 10, 16)),
    TemporalFold("F2", date(2025, 11, 15)),
    TemporalFold("F3", date(2025, 12, 15)),
    TemporalFold("F4", date(2026, 1, 14)),
)

PROTOCOLS = {"FOUR_FOLD_250K_V1": FOUR_FOLD_250K_V1}


def get_protocol(name: str) -> tuple[TemporalFold, ...]:
    try:
        return PROTOCOLS[name]
    except KeyError as error:
        raise ValueError(f"Unknown direct temporal protocol: {name}") from error
