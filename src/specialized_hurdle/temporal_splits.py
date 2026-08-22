"""Temporal Split Generator for Specialized Hurdle Stack.

Strict Time-CV rule:
For a validation/OOF anchor V, an anchor A is a valid training anchor if and only if:
    A + 30 days <= V
Zero target overlap between training targets and validation target.
"""

from datetime import datetime, timedelta
from typing import List
from src.specialized_hurdle.definitions import (
    ALL_AVAILABLE_ANCHORS,
    JANUARY_VALIDATION_ANCHOR,
    TemporalFold,
)


def get_legal_train_anchors(outer_anchor: str, all_anchors: List[str] = ALL_AVAILABLE_ANCHORS) -> List[str]:
    v_dt = datetime.strptime(outer_anchor, "%Y-%m-%d").date()
    legal = []
    for a in all_anchors:
        a_dt = datetime.strptime(a, "%Y-%m-%d").date()
        # Train target window [A + 1, A + 30] must end at or before outer anchor V
        if a_dt + timedelta(days=30) <= v_dt:
            legal.append(a)
    return legal


def build_meta_oof_folds(
    candidate_outer_anchors: List[str] = None,
    all_anchors: List[str] = ALL_AVAILABLE_ANCHORS,
    min_train_anchors: int = 5,
) -> List[TemporalFold]:
    """Generates the 4 expanding-window Meta-OOF temporal folds ending before January 2026-01-14."""
    if candidate_outer_anchors is None:
        # 4 expanding-window outer anchors before January 2026-01-14:
        # '2025-10-27', '2025-11-24', '2025-12-15', '2026-01-14'
        candidate_outer_anchors = ["2025-10-27", "2025-11-24", "2025-12-15", "2026-01-14"]

    folds = []
    for idx, outer_anchor in enumerate(candidate_outer_anchors):
        train_anchors = get_legal_train_anchors(outer_anchor, all_anchors)
        if len(train_anchors) < min_train_anchors and outer_anchor != candidate_outer_anchors[0]:
            continue

        # Inner validation is the latest legal train anchor
        inner_val = train_anchors[-1] if len(train_anchors) > 0 else outer_anchor
        inner_train = get_legal_train_anchors(inner_val, all_anchors)

        fold = TemporalFold(
            fold_id=f"fold_{idx:02d}",
            outer_anchor=outer_anchor,
            train_anchors=train_anchors,
            inner_val_anchor=inner_val,
            inner_train_anchors=inner_train,
            n_train_anchors=len(train_anchors),
        )
        folds.append(fold)

    return folds
