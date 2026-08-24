"""Cross-validation orchestration skeleton.

The filling agent must build/release one fold at a time and never reuse fitted
models, optimizer state, or validation labels between folds.
"""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig


def run_cross_validation(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> None:
    """TODO: execute each 250k fold and write immutable prediction banks."""
    raise NotImplementedError("Fill direct temporal CV orchestration")


def run_final_submission(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> Path:
    """TODO: fresh final training after a separately approved CV result."""
    raise NotImplementedError("Final submission is intentionally not implemented in the skeleton")
