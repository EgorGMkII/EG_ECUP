from __future__ import annotations

import pytest

from src.ssl_temporal_stack_v1.training import (
    round_robin_batches,
    train_exact_accumulated_steps,
    train_exact_steps,
)


def test_exact_steps_restart_anchor_iterators_and_cover_every_anchor() -> None:
    anchors = ("a", "b", "c")

    def factory(anchor: str):
        def build(cycle: int):
            yield (f"{anchor}-{cycle}", 4)
            yield (f"{anchor}-{cycle}", 3)
        return build

    stream = round_robin_batches({anchor: factory(anchor) for anchor in anchors})
    stats = train_exact_steps(batches=stream, requested_steps=17, train_step=lambda _: 0.5)
    assert stats.completed_steps == 17
    assert stats.examples_seen == 60
    assert stats.per_anchor_batches == {"a": 6, "b": 6, "c": 5}


def test_empty_anchor_fails_fast() -> None:
    stream = round_robin_batches({"empty": lambda cycle: iter(())})
    with pytest.raises(RuntimeError, match="produced no batches"):
        train_exact_steps(batches=stream, requested_steps=1, train_step=lambda _: 0.0)


def test_finite_stream_cannot_undercomplete_budget() -> None:
    from src.ssl_temporal_stack_v1.training import AnchoredBatch

    stream = iter([AnchoredBatch("a", 1, 1)])
    with pytest.raises(RuntimeError, match="ended before exact step budget"):
        train_exact_steps(batches=stream, requested_steps=2, train_step=lambda _: 0.0)


def test_accumulation_counts_optimizer_steps_not_micro_batches() -> None:
    stream = round_robin_batches({
        "a": lambda cycle: iter([("a", 2), ("a", 2)]),
        "b": lambda cycle: iter([("b", 3), ("b", 3)]),
    })
    optimizer_calls: list[int] = []
    stats = train_exact_accumulated_steps(
        batches=stream,
        requested_steps=3,
        accumulation_steps=4,
        micro_step=lambda value, divisor: 4.0 / divisor,
        optimizer_step=lambda: optimizer_calls.append(1),
    )
    assert stats.completed_steps == 3
    assert len(optimizer_calls) == 3
    assert sum(stats.per_anchor_batches.values()) == 12
    assert stats.examples_seen == 30
