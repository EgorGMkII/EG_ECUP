"""Canonical Definitions and Types for Specialized Hurdle Stack."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class TransitionState(str, Enum):
    INACTIVE_TO_INACTIVE = "0->0"
    INACTIVE_TO_ACTIVE = "0->1"
    ACTIVE_TO_INACTIVE = "1->0"
    ACTIVE_TO_ACTIVE = "1->1"


@dataclass(frozen=True)
class TemporalFold:
    fold_id: str
    outer_anchor: str
    train_anchors: List[str]
    inner_val_anchor: str
    inner_train_anchors: List[str]
    n_train_anchors: int


# Canonical Anchor dates for Purged Time-CV
ALL_AVAILABLE_ANCHORS = [
    "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
    "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
    "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
    "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15",
    "2025-12-22", "2026-01-05", "2026-01-14"
]

JANUARY_VALIDATION_ANCHOR = "2026-01-14"
TEST_ANCHOR = "2026-02-13"


def compute_transition_state(was_active: bool, will_buy: bool) -> TransitionState:
    if not was_active and not will_buy:
        return TransitionState.INACTIVE_TO_INACTIVE
    elif not was_active and will_buy:
        return TransitionState.INACTIVE_TO_ACTIVE
    elif was_active and not will_buy:
        return TransitionState.ACTIVE_TO_INACTIVE
    else:
        return TransitionState.ACTIVE_TO_ACTIVE
