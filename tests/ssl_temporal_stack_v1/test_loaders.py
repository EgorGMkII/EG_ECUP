from __future__ import annotations

import numpy as np

from src.ssl_temporal_stack_v1.loaders import IndexedArrays, make_loader_factory


def test_indexed_arrays_applies_task_mask_without_copying_source() -> None:
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)
    dataset = IndexedArrays((x, y), np.array([1, 4, 8]))
    assert len(dataset) == 3
    assert dataset[1][0].tolist() == [8.0, 9.0]
    assert float(dataset[1][1]) == 4.0


def test_loader_shuffle_is_deterministic_per_cycle() -> None:
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    factory = make_loader_factory(
        arrays=(x,), indices=None, batch_size=4,
        seed_parts=("RUN_A", "s1", "ssl", "anchor"), workers=0,
    )
    first = [batch[0][0][:, 0].tolist() for batch in factory(0)]
    repeated = [batch[0][0][:, 0].tolist() for batch in factory(0)]
    next_cycle = [batch[0][0][:, 0].tolist() for batch in factory(1)]
    assert first == repeated
    assert first != next_cycle


def test_loader_reports_partial_final_batch_size() -> None:
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    factory = make_loader_factory(
        arrays=(x,), indices=None, batch_size=4,
        seed_parts=("RUN_A", "s1", "base", "anchor"), workers=0,
    )
    batches = list(factory(0))
    assert [examples for _, examples in batches] == [4, 1]
