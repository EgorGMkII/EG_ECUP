"""Versioned temporal profiles used by reference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TemporalProfile:
    name: str
    run_a_anchors: tuple[str, ...]
    meta_anchor: str
    run_b_anchors: tuple[str, ...]
    validation_anchor: str
    final_train_anchors: tuple[str, ...]
    final_inference_anchor: str

    @property
    def validation_target_start(self) -> str:
        return (date.fromisoformat(self.validation_anchor) + timedelta(days=1)).isoformat()

    @property
    def validation_target_end(self) -> str:
        return (date.fromisoformat(self.validation_anchor) + timedelta(days=30)).isoformat()


_RUN_A = (
    "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04", "2025-08-18",
    "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13", "2025-10-27",
    "2025-11-10",
)
_RUN_B = _RUN_A + ("2025-11-24", "2025-12-08", "2025-12-15")

POST_NY_PUBLIC_PROXY = TemporalProfile(
    name="POST_NY_PUBLIC_PROXY",
    run_a_anchors=_RUN_A,
    meta_anchor="2025-12-15",
    run_b_anchors=_RUN_B,
    validation_anchor="2026-01-14",
    final_train_anchors=(
        "2025-08-04", "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29",
        "2025-10-13", "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08",
        "2025-12-15", "2025-12-22", "2026-01-05", "2026-01-14",
    ),
    final_inference_anchor="2026-02-13",
)

PROFILES = {POST_NY_PUBLIC_PROXY.name: POST_NY_PUBLIC_PROXY}


def get_profile(name: str) -> TemporalProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown temporal profile: {name}") from error
