"""Resolve config epochs to exact optimizer steps after train stores exist."""

from __future__ import annotations

from copy import deepcopy
from math import ceil
from typing import Any, Mapping

import numpy as np

from .base import RunContext


def _steps(samples: int, effective_batch_size: int, epochs: int) -> int:
    if samples <= 0:
        raise ValueError("An epoch cannot be resolved for an empty training subset")
    if effective_batch_size <= 0 or epochs < 1:
        raise ValueError("Invalid epoch budget")
    return ceil(samples / effective_batch_size) * epochs


def _stage(raw: Mapping[str, Any], *, samples: int, batches_per_epoch: int, effective_batch_size: int) -> dict[str, Any]:
    result = dict(raw)
    if "epochs" not in result:
        # Frozen step recipes remain supported and stay bit-for-bit unchanged.
        return result
    epochs = int(result.pop("epochs"))
    if batches_per_epoch <= 0:
        raise ValueError("An epoch must contain at least one batch")
    steps = batches_per_epoch * epochs
    warmup_fraction = float(result.pop("warmup_fraction", 0.0))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    result["steps"] = steps
    result["warmup_steps"] = int(round(steps * warmup_fraction))
    result["epoch_resolution"] = {
        "requested_epochs": epochs,
        "samples_per_epoch": samples,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": batches_per_epoch,
        "requested_steps": steps,
        "warmup_fraction": warmup_fraction,
    }
    return result


def _anchor_samples(context: RunContext, task: str | None) -> list[int]:
    counts: list[int] = []
    for anchor in context.train_anchors:
        frame = context.stores.frames.get(anchor)
        if task is None:
            counts.append(frame.height)
        elif task == "react":
            counts.append(int((frame["was_active"].to_numpy() == 0).sum()))
        elif task == "churn":
            counts.append(int((frame["was_active"].to_numpy() == 1).sum()))
        elif task == "amount":
            counts.append(int((frame["z_target"].to_numpy() > 0).sum()))
        else:
            raise ValueError(f"Unknown specialist task: {task}")
    return counts


def resolve_epoch_recipe(context: RunContext, model_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a run-specific, fully step-resolved copy of one neural recipe."""
    result = deepcopy(dict(raw))
    if model_id in {"catboost", "catboost_direct"}:
        return result
    if model_id in {"s1", "s2", "tcn", "residual_mlp"}:
        effective_batch = int(result["batch_size"])
        loader_batch = effective_batch
        accumulation = 1
    elif model_id == "ett":
        loader_batch = int(result["micro_batch_size"])
        accumulation = int(result["accumulation_steps"])
        effective_batch = loader_batch * accumulation
    else:
        raise ValueError(f"Unknown model for epoch resolution: {model_id}")
    def stage(raw_stage: Mapping[str, Any], task: str | None) -> dict[str, Any]:
        counts = _anchor_samples(context, task)
        samples = sum(counts)
        micro_batches = sum(ceil(count / loader_batch) for count in counts if count > 0)
        # ETT consumes accumulation micro-batches per optimizer update. A
        # final incomplete group is rounded up; at most accumulation-1 rows of
        # the next epoch are reused, and the report makes that explicit.
        optimizer_batches = ceil(micro_batches / accumulation)
        return _stage(raw_stage, samples=samples, batches_per_epoch=optimizer_batches, effective_batch_size=effective_batch)

    if "ssl" in result and isinstance(result["ssl"], Mapping) and int(result["ssl"].get("steps", result["ssl"].get("epochs", 1))) != 0:
        result["ssl"] = stage(result["ssl"], None)
    result["base"] = stage(result["base"], None)
    for task, phases in result["specialists"].items():
        result["specialists"][task] = {
            phase: stage(value, task)
            for phase, value in phases.items()
        }
    return result


def epoch_resolution_report(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Extract resolved epoch metadata without polluting trainer optimizer inputs."""
    report: dict[str, Any] = {}
    for stage in ("ssl", "base"):
        if isinstance(recipe.get(stage), Mapping) and "epoch_resolution" in recipe[stage]:
            report[stage] = recipe[stage]["epoch_resolution"]
    specialists: dict[str, Any] = {}
    for task, phases in recipe.get("specialists", {}).items():
        details = {phase: value["epoch_resolution"] for phase, value in phases.items() if "epoch_resolution" in value}
        if details:
            specialists[task] = details
    if specialists:
        report["specialists"] = specialists
    return report
