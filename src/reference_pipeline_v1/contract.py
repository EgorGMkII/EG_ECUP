"""Immutable contract and temporal integrity checks for the V1 baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

ROOT_SEED = 42
COHORT_SHA256 = "d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2"
CHANNELS = (
    "searches", "to_cart", "to_ord", "gmv", "search_to_cart", "search_to_ord",
    "cat_to_cart", "cat_to_ord", "gmv_search", "gmv_cat", "is_active",
    "is_purchase_day", "sin_dow", "cos_dow", "normalized_position",
)
MODEL_ORDER = ("CatBoost", "S1", "S2", "ETT")


@dataclass(frozen=True)
class Profile:
    name: str
    run_a_anchors: tuple[str, ...]
    meta_anchor: str
    run_b_anchors: tuple[str, ...]
    validation_anchor: str


POST_NY_PUBLIC_PROXY = Profile(
    name="POST_NY_PUBLIC_PROXY",
    run_a_anchors=("2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26", "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04", "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13", "2025-10-27", "2025-11-10"),
    meta_anchor="2025-12-15",
    run_b_anchors=("2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26", "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04", "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13", "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15"),
    validation_anchor="2026-01-14",
)


def derive_seed(run: str, architecture: str, task: str) -> int:
    """Stable seed derived exclusively from the fixed root seed and identity."""
    raw = f"{ROOT_SEED}|{run}|{architecture}|{task}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def cohort_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def windows(anchor: str) -> dict[str, str]:
    value = date.fromisoformat(anchor)
    return {
        "anchor": anchor,
        "state_history_start": str(value - timedelta(days=89)),
        "state_history_end": anchor,
        "model_target_start": str(value + timedelta(days=1)),
        "model_target_end": str(value + timedelta(days=30)),
        "s1_s2_history_start": str(value - timedelta(days=179)),
        "ett_history_start": str(value - timedelta(days=364)),
    }


def validate_profile(profile: Profile = POST_NY_PUBLIC_PROXY) -> None:
    assert profile.meta_anchor not in profile.run_a_anchors
    assert profile.validation_anchor not in profile.run_b_anchors
    assert max(date.fromisoformat(windows(x)["model_target_end"]) for x in profile.run_a_anchors) <= date.fromisoformat(profile.meta_anchor)
    assert max(date.fromisoformat(windows(x)["model_target_end"]) for x in profile.run_b_anchors) <= date.fromisoformat(profile.validation_anchor)


def anchor_manifest(profile: Profile = POST_NY_PUBLIC_PROXY) -> list[dict[str, str]]:
    validate_profile(profile)
    rows: list[dict[str, str]] = []
    for run, anchors, holdout in (("RUN_A", profile.run_a_anchors, profile.meta_anchor), ("RUN_B", profile.run_b_anchors, profile.validation_anchor)):
        rows.extend({"run": run, "role": "train", **windows(anchor)} for anchor in anchors)
        rows.append({"run": run, "role": "holdout", **windows(holdout)})
    return rows
