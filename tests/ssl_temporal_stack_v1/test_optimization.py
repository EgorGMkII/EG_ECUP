from __future__ import annotations

import pytest
import torch

from src.ssl_temporal_stack_v1.optimization import StepOptimizer


def test_step_optimizer_counts_real_updates_and_completes() -> None:
    model = torch.nn.Linear(2, 1)
    control = StepOptimizer(
        model.parameters(), total_steps=3, learning_rate=1e-3,
        weight_decay=0.0, warmup_steps=1, device=torch.device("cpu"),
    )
    rates = []
    for _ in range(3):
        control.prepare()
        rates.append(control.current_lr)
        control.backward(model(torch.ones(4, 2)).square().mean())
        control.finish()
    control.assert_complete()
    assert control.completed_steps == 3
    assert rates[0] == pytest.approx(1e-3)
    assert rates[-1] == pytest.approx(0.0)


def test_step_optimizer_rejects_undercompletion() -> None:
    model = torch.nn.Linear(2, 1)
    control = StepOptimizer(
        model.parameters(), total_steps=2, learning_rate=1e-3,
        weight_decay=0.0, warmup_steps=0, device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="Completed 0 of 2"):
        control.assert_complete()


def test_step_optimizer_rejects_non_finite_loss() -> None:
    model = torch.nn.Linear(2, 1)
    control = StepOptimizer(
        model.parameters(), total_steps=1, learning_rate=1e-3,
        weight_decay=0.0, warmup_steps=0, device=torch.device("cpu"),
    )
    control.prepare()
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        control.backward(torch.tensor(float("nan"), requires_grad=True))
