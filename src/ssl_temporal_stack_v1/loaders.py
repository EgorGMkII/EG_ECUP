"""Deterministic task-specific anchor loaders for SSL V1."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .contract import EXPERIMENT
from .runtime import derive_seed
from .stores import StoreRegistry
from .training import BatchFactory


class IndexedArrays(Dataset[tuple[Any, ...]]):
    """Zero-copy index view over aligned NumPy arrays."""

    def __init__(self, arrays: Sequence[np.ndarray], indices: np.ndarray | None = None) -> None:
        if not arrays:
            raise ValueError("At least one array is required")
        size = len(arrays[0])
        if any(len(array) != size for array in arrays):
            raise ValueError("Loader arrays are not aligned")
        self.arrays = tuple(arrays)
        self.indices = np.arange(size, dtype=np.int64) if indices is None else indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, ...]:
        index = int(self.indices[position])
        return tuple(array[index] for array in self.arrays)


def make_loader_factory(
    *,
    arrays: Sequence[np.ndarray],
    indices: np.ndarray | None,
    batch_size: int,
    seed_parts: tuple[object, ...],
    shuffle: bool = True,
    workers: int | None = None,
) -> BatchFactory[tuple[torch.Tensor, ...]]:
    dataset = IndexedArrays(arrays, indices)
    if len(dataset) == 0:
        def empty(_: int) -> Iterator[tuple[tuple[torch.Tensor, ...], int]]:
            return iter(())
        return empty
    if workers is None:
        workers = 2 if os.name != "nt" and torch.cuda.is_available() else 0

    def factory(cycle: int) -> Iterator[tuple[tuple[torch.Tensor, ...], int]]:
        generator = torch.Generator()
        generator.manual_seed(derive_seed(EXPERIMENT.root_seed, *seed_parts, cycle))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=False,
        )
        for batch in loader:
            tensors = tuple(batch)
            yield tensors, int(tensors[0].shape[0])

    return factory


def _float32(values: Any) -> np.ndarray:
    return values.to_numpy().astype(np.float32, copy=False)


def dense_ssl_factories(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    batch_size: int,
) -> dict[str, BatchFactory[tuple[torch.Tensor, ...]]]:
    if model_id not in {"s1", "s2"}:
        raise ValueError(f"Dense SSL is unsupported for {model_id}")
    result: dict[str, BatchFactory[tuple[torch.Tensor, ...]]] = {}
    for anchor in anchors:
        daily = stores.daily.get(anchor)
        arrays: tuple[np.ndarray, ...]
        if model_id == "s1":
            arrays = (daily,)
        else:
            horizons = stores.horizons.get(anchor)
            arrays = (daily, horizons[7], horizons[14], horizons[30])
        result[anchor] = make_loader_factory(
            arrays=arrays,
            indices=None,
            batch_size=batch_size,
            seed_parts=(run, model_id, "ssl", anchor),
        )
    return result

def dense_base_factories(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    batch_size: int,
) -> dict[str, BatchFactory[tuple[torch.Tensor, ...]]]:
    result: dict[str, BatchFactory[tuple[torch.Tensor, ...]]] = {}
    for anchor in anchors:
        frame = stores.frames.get(anchor)
        arrays = (
            stores.daily.get(anchor), _float32(frame["z_target"]),
            _float32(frame["was_active"]), _float32(frame["will_buy"]),
        )
        result[anchor] = make_loader_factory(
            arrays=arrays,
            indices=None,
            batch_size=batch_size,
            seed_parts=(run, model_id, "base", anchor),
        )
    return result


def dense_specialist_factories(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    task: str,
    phase: str,
    batch_size: int,
) -> dict[str, BatchFactory[tuple[torch.Tensor, ...]]]:
    if task not in {"react", "churn", "amount"}:
        raise ValueError(f"Unknown specialist task: {task}")
    result: dict[str, BatchFactory[tuple[torch.Tensor, ...]]] = {}
    for anchor in anchors:
        frame = stores.frames.get(anchor)
        active = _float32(frame["was_active"])
        buy = _float32(frame["will_buy"])
        if task == "react":
            indices = np.flatnonzero(active == 0)
            target = buy
        elif task == "churn":
            indices = np.flatnonzero(active == 1)
            target = 1.0 - buy
        else:
            target = _float32(frame["z_target"])
            indices = np.flatnonzero(target > 0)
        result[anchor] = make_loader_factory(
            arrays=(stores.daily.get(anchor), target),
            indices=indices,
            batch_size=batch_size,
            seed_parts=(run, model_id, "specialist", task, phase, anchor),
        )
    return result


def event_base_factories(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    batch_size: int,
) -> dict[str, BatchFactory[tuple[torch.Tensor, ...]]]:
    result: dict[str, BatchFactory[tuple[torch.Tensor, ...]]] = {}
    for anchor in anchors:
        frame = stores.frames.get(anchor)
        arrays = (
            *stores.events.get(anchor), _float32(frame["z_target"]),
            _float32(frame["was_active"]), _float32(frame["will_buy"]),
        )
        result[anchor] = make_loader_factory(
            arrays=arrays,
            indices=None,
            batch_size=batch_size,
            seed_parts=(run, "ett", "base", anchor),
        )
    return result


def event_specialist_factories(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    task: str,
    phase: str,
    batch_size: int,
) -> dict[str, BatchFactory[tuple[torch.Tensor, ...]]]:
    if task not in {"react", "churn", "amount"}:
        raise ValueError(f"Unknown specialist task: {task}")
    result: dict[str, BatchFactory[tuple[torch.Tensor, ...]]] = {}
    for anchor in anchors:
        frame = stores.frames.get(anchor)
        active = _float32(frame["was_active"])
        buy = _float32(frame["will_buy"])
        if task == "react":
            indices = np.flatnonzero(active == 0)
            target = buy
        elif task == "churn":
            indices = np.flatnonzero(active == 1)
            target = 1.0 - buy
        else:
            target = _float32(frame["z_target"])
            indices = np.flatnonzero(target > 0)
        result[anchor] = make_loader_factory(
            arrays=(*stores.events.get(anchor), target),
            indices=indices,
            batch_size=batch_size,
            seed_parts=(run, "ett", "specialist", task, phase, anchor),
        )
    return result
