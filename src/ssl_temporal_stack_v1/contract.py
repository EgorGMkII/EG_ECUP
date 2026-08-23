"""Immutable contract for SSL_TEMPORAL_STACK_V1.

This module deliberately does not import the retired reference pipeline.  A
change to any constant here requires a new versioned experiment ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path


ROOT_SEED = 42
COHORT_SHA256 = "d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2"
FEATURE_ORDER_PATH = Path("configs/ssl_temporal_stack_v1/catboost_feature_order.json")
FEATURE_ORDER_SHA256 = "e54a385cd69665f09063762a2e8581fe953b759004a69180487dfcea3fa8df6e"

RUN_A_ANCHORS = (
    "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
    "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29",
    "2025-10-13", "2025-10-27", "2025-11-10",
)
META_ANCHOR = "2025-12-15"
RUN_B_ANCHORS = RUN_A_ANCHORS + ("2025-11-24", "2025-12-08", META_ANCHOR)
VALIDATION_ANCHOR = "2026-01-14"


@dataclass(frozen=True)
class NeuralBudget:
    ssl_steps: int
    base_steps: int
    specialist_head_steps: int
    specialist_finetune_steps: int
    batch_size: int = 512

    @property
    def total_steps(self) -> int:
        return self.ssl_steps + self.base_steps + 3 * (
            self.specialist_head_steps + self.specialist_finetune_steps
        )


@dataclass(frozen=True)
class SSLTemporalExperiment:
    experiment_id: str
    root_seed: int
    run_a_anchors: tuple[str, ...]
    meta_anchor: str
    run_b_anchors: tuple[str, ...]
    validation_anchor: str
    model_ids: tuple[str, ...]
    budgets: dict[str, NeuralBudget]
    cohort_sha256: str
    feature_order_path: Path
    feature_order_sha256: str


EXPERIMENT = SSLTemporalExperiment(
    experiment_id="SSL_TEMPORAL_STACK_V1",
    root_seed=ROOT_SEED,
    run_a_anchors=RUN_A_ANCHORS,
    meta_anchor=META_ANCHOR,
    run_b_anchors=RUN_B_ANCHORS,
    validation_anchor=VALIDATION_ANCHOR,
    model_ids=("catboost", "s1", "s2", "ett"),
    budgets={
        "s1": NeuralBudget(750, 2000, 400, 600),
        "s2": NeuralBudget(750, 2000, 400, 600),
        "ett": NeuralBudget(0, 3500, 400, 600),
    },
    cohort_sha256=COHORT_SHA256,
    feature_order_path=FEATURE_ORDER_PATH,
    feature_order_sha256=FEATURE_ORDER_SHA256,
)


def _target_end(anchor: str) -> date:
    return date.fromisoformat(anchor) + timedelta(days=30)


def load_feature_order(path: Path = FEATURE_ORDER_PATH) -> tuple[str, ...]:
    package = json.loads(path.read_text(encoding="utf-8"))
    features = tuple(package["ordered_features"])
    encoded = json.dumps(features, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if package.get("count") != len(features):
        raise ValueError("CatBoost feature count does not match ordered list")
    if package.get("ordered_features_sha256") != digest:
        raise ValueError("CatBoost feature package contains an invalid SHA256")
    return features


def validate_experiment(config: SSLTemporalExperiment = EXPERIMENT) -> None:
    if config.experiment_id != "SSL_TEMPORAL_STACK_V1":
        raise ValueError("Unexpected experiment ID")
    if len(config.run_a_anchors) != 11 or len(config.run_b_anchors) != 14:
        raise ValueError("SSL V1 requires exactly 11 RUN A and 14 RUN B anchors")
    if config.run_b_anchors[: len(config.run_a_anchors)] != config.run_a_anchors:
        raise ValueError("RUN A anchors must be the chronological prefix of RUN B")
    if tuple(sorted(config.run_a_anchors)) != config.run_a_anchors:
        raise ValueError("RUN A anchors are not chronological")
    if tuple(sorted(config.run_b_anchors)) != config.run_b_anchors:
        raise ValueError("RUN B anchors are not chronological")
    if max(map(_target_end, config.run_a_anchors)) > date.fromisoformat(config.meta_anchor):
        raise ValueError("RUN A target crosses M")
    if max(map(_target_end, config.run_b_anchors)) > date.fromisoformat(config.validation_anchor):
        raise ValueError("RUN B target crosses V")
    if _target_end(config.run_b_anchors[-1]) != date.fromisoformat(config.validation_anchor):
        raise ValueError("Last RUN B target must end exactly on V")
    expected = {
        "s1": NeuralBudget(750, 2000, 400, 600),
        "s2": NeuralBudget(750, 2000, 400, 600),
        "ett": NeuralBudget(0, 3500, 400, 600),
    }
    if config.budgets != expected:
        raise ValueError("Neural step budgets differ from frozen SSL V1 contract")
    if sum(budget.total_steps for budget in config.budgets.values()) != 18_000:
        raise ValueError("Per-run optimizer-step total must equal 18000")
    features = load_feature_order(config.feature_order_path)
    encoded = json.dumps(features, separators=(",", ":")).encode("utf-8")
    if len(features) != 374 or hashlib.sha256(encoded).hexdigest() != config.feature_order_sha256:
        raise ValueError("CatBoost feature contract differs from frozen SSL V1")
