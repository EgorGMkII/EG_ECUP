from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.ssl_temporal_stack_v1.contract import NeuralBudget
from src.ssl_temporal_stack_v1.trainers import fit_gru_pretrainer


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
