"""Exact-step, anchor-balanced training primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import time
from typing import Generic, TypeVar


Batch = TypeVar("Batch")


@dataclass(frozen=True)
class AnchoredBatch(Generic[Batch]):
    anchor: str
    value: Batch
    examples: int


@dataclass(frozen=True)
class TrainingStats:
    requested_steps: int
    completed_steps: int
    examples_seen: int
    per_anchor_batches: dict[str, int]
    elapsed_seconds: float
    final_loss: float


BatchFactory = Callable[[int], Iterator[tuple[Batch, int]]]


def round_robin_batches(
    factories: Mapping[str, BatchFactory[Batch]],
) -> Iterator[AnchoredBatch[Batch]]:
    """Yield one batch per anchor in stable order, recreating exhausted iterators.

    Each factory receives its zero-based iterator cycle.  The caller uses that
    cycle in the derived shuffle seed, making restarts deterministic.
    """

    if not factories:
        raise ValueError("At least one anchor batch factory is required")
    anchors = tuple(factories)
    cycles = {anchor: 0 for anchor in anchors}
    iterators = {anchor: iter(factories[anchor](0)) for anchor in anchors}
    while True:
        for anchor in anchors:
            try:
                value, examples = next(iterators[anchor])
            except StopIteration:
                cycles[anchor] += 1
                iterator = iter(factories[anchor](cycles[anchor]))
                iterators[anchor] = iterator
                try:
                    value, examples = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(f"Anchor {anchor} produced no batches") from error
            if examples <= 0:
                raise RuntimeError(f"Anchor {anchor} produced an empty batch")
            yield AnchoredBatch(anchor=anchor, value=value, examples=examples)


def train_exact_steps(
    *,
    batches: Iterator[AnchoredBatch[Batch]],
    requested_steps: int,
    train_step: Callable[[Batch], float],
    progress_every: int = 250,
    progress: Callable[[int, int, str, float], None] | None = None,
) -> TrainingStats:
    """Run exactly ``requested_steps`` optimizer steps or fail."""

    if requested_steps <= 0:
        raise ValueError("requested_steps must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    started = time.perf_counter()
    counts: dict[str, int] = {}
    examples_seen = 0
    final_loss = float("nan")
    completed = 0
    while completed < requested_steps:
        try:
            batch = next(batches)
        except StopIteration as error:
            raise RuntimeError("Batch stream ended before exact step budget") from error
        final_loss = float(train_step(batch.value))
        completed += 1
        examples_seen += batch.examples
        counts[batch.anchor] = counts.get(batch.anchor, 0) + 1
        if progress is not None and (completed % progress_every == 0 or completed == requested_steps):
            progress(completed, requested_steps, batch.anchor, final_loss)
    if completed != requested_steps:
        raise RuntimeError(f"Completed {completed} of {requested_steps} optimizer steps")
    return TrainingStats(
        requested_steps=requested_steps,
        completed_steps=completed,
        examples_seen=examples_seen,
        per_anchor_batches=counts,
        elapsed_seconds=time.perf_counter() - started,
        final_loss=final_loss,
    )


def train_exact_accumulated_steps(
    *,
    batches: Iterator[AnchoredBatch[Batch]],
    requested_steps: int,
    accumulation_steps: int,
    micro_step: Callable[[Batch, int], float],
    optimizer_step: Callable[[], None],
    progress_every: int = 250,
    progress: Callable[[int, int, str, float], None] | None = None,
) -> TrainingStats:
    """Run exact optimizer steps with a fixed number of micro-batches per step."""

    if requested_steps <= 0:
        raise ValueError("requested_steps must be positive")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    started = time.perf_counter()
    counts: dict[str, int] = {}
    examples_seen = 0
    final_loss = float("nan")
    completed = 0
    while completed < requested_steps:
        losses: list[float] = []
        last_anchor = ""
        for micro_index in range(accumulation_steps):
            try:
                batch = next(batches)
            except StopIteration as error:
                raise RuntimeError("Batch stream ended during gradient accumulation") from error
            losses.append(float(micro_step(batch.value, accumulation_steps)))
            examples_seen += batch.examples
            counts[batch.anchor] = counts.get(batch.anchor, 0) + 1
            last_anchor = batch.anchor
        optimizer_step()
        completed += 1
        final_loss = sum(losses) / len(losses)
        if progress is not None and (completed % progress_every == 0 or completed == requested_steps):
            progress(completed, requested_steps, last_anchor, final_loss)
    if completed != requested_steps:
        raise RuntimeError(f"Completed {completed} of {requested_steps} optimizer steps")
    return TrainingStats(
        requested_steps=requested_steps,
        completed_steps=completed,
        examples_seen=examples_seen,
        per_anchor_batches=counts,
        elapsed_seconds=time.perf_counter() - started,
        final_loss=final_loss,
    )
