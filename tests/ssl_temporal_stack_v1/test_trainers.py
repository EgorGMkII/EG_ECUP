from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.ssl_temporal_stack_v1.contract import NeuralBudget
from src.ssl_temporal_stack_v1 import trainers
from src.ssl_temporal_stack_v1.trainers import fit_gru_pretrainer
from src.ssl_temporal_stack_v1.training import AnchoredBatch


class TinyDaily:
    def get(self, anchor: str) -> np.ndarray:
        return np.zeros((4, 180, 15), dtype=np.float32)


class TinyHorizons:
    def get(self, anchor: str) -> dict[int, np.ndarray]:
        return {horizon: np.arange(4, dtype=np.float32) for horizon in (7, 14, 30)}


def test_s1_ssl_smoke_completes_exact_budget_on_cpu() -> None:
    stores = SimpleNamespace(daily=TinyDaily(), horizons=TinyHorizons())
    budget = NeuralBudget(ssl_steps=1, base_steps=1, specialist_head_steps=1, specialist_finetune_steps=1, batch_size=2)
    _, stats = fit_gru_pretrainer(
        stores, ("2025-06-23",), run="SMOKE", model_id="s1",
        device=torch.device("cpu"), budget=budget,
    )
    assert stats.completed_steps == 1
    assert stats.examples_seen == 2


def test_s2_ssl_smoke_completes_exact_budget_on_cpu() -> None:
    stores = SimpleNamespace(daily=TinyDaily(), horizons=TinyHorizons())
    budget = NeuralBudget(ssl_steps=1, base_steps=1, specialist_head_steps=1, specialist_finetune_steps=1, batch_size=2)
    _, stats = fit_gru_pretrainer(
        stores, ("2025-06-23",), run="SMOKE", model_id="s2",
        device=torch.device("cpu"), budget=budget,
    )
    assert stats.completed_steps == 1
    assert stats.examples_seen == 2


def test_ssl_disabled_constructs_default_pretrainer_without_recipe() -> None:
    stores = SimpleNamespace(daily=TinyDaily(), horizons=TinyHorizons())
    budget = NeuralBudget(ssl_steps=0, base_steps=1, specialist_head_steps=1, specialist_finetune_steps=1, batch_size=2)
    _, stats = fit_gru_pretrainer(
        stores, ("2025-06-23",), run="SMOKE", model_id="s1",
        device=torch.device("cpu"), budget=budget,
    )
    assert stats.completed_steps == 0


class _TinyDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0] * self.scale


class _TinyEvent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, content: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        return content[:, 0, 0] * self.scale


def test_specialist_phases_accept_optional_recipe_and_complete_one_step(monkeypatch) -> None:
    stores = SimpleNamespace(anchor_tickets=None)
    dense_batch = AnchoredBatch("2025-06-23", (torch.ones((2, 2)), torch.ones(2)), 2)
    monkeypatch.setattr(trainers, "dense_specialist_factories", lambda *args, **kwargs: {"2025-06-23": None})
    monkeypatch.setattr(trainers, "round_robin_batches", lambda *args, **kwargs: iter([dense_batch]))
    dense = trainers._fit_dense_specialist_phase(
        _TinyDense(), stores, ("2025-06-23",), run="SMOKE", model_id="s1",
        task="react", phase="H", steps=1, learning_rate=1e-3, batch_size=2,
        device=torch.device("cpu"), recipe=None,
    )
    assert dense.completed_steps == 1

    event_batch = AnchoredBatch(
        "2025-06-23",
        (torch.ones((2, 1, 1)), torch.ones((2, 1, 1)), torch.zeros((2, 1), dtype=torch.long), torch.zeros((2, 1), dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.ones(2)),
        2,
    )
    monkeypatch.setattr(trainers, "event_specialist_factories", lambda *args, **kwargs: {"2025-06-23": None})
    monkeypatch.setattr(trainers, "round_robin_batches", lambda *args, **kwargs: iter([event_batch]))
    event = trainers._fit_ett_specialist_phase(
        _TinyEvent(), stores, ("2025-06-23",), run="SMOKE", task="react",
        phase="H", steps=1, learning_rate=1e-3, device=torch.device("cpu"),
        micro_batch_size=2, accumulation_steps=1, recipe=None,
    )
    assert event.completed_steps == 1
