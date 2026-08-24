"""Exact-step neural trainers shared identically by RUN A and RUN B."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contract import EXPERIMENT, NeuralBudget
from .loaders import (
    dense_base_factories,
    dense_specialist_factories,
    dense_ssl_factories,
    event_base_factories,
    event_specialist_factories,
)
from .models import (
    EventTimeTransformer,
    S1MaskedPretrainer,
    S2MultiHorizonPretrainer,
    Specialist,
    TransitionBase,
    transition_loss,
)
from .optimization import StepOptimizer
from .recipes import NeuralRecipe, OptimizerRecipe
from .runtime import progress, seed_everything
from .stores import StoreRegistry
from .training import (
    TrainingStats,
    round_robin_batches,
    train_exact_accumulated_steps,
    train_exact_steps,
)


Pretrainer = S1MaskedPretrainer | S2MultiHorizonPretrainer


def _zero_stats() -> TrainingStats:
    return TrainingStats(0, 0, 0, {}, 0.0, float("nan"))


@dataclass
class BaseFit:
    model: nn.Module
    stats: TrainingStats


@dataclass
class SpecialistFit:
    model: Specialist
    stats: dict[str, TrainingStats]


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _to_device(tensor: torch.Tensor, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = tensor.to(device, non_blocking=device.type == "cuda")
    return result.to(dtype=dtype) if dtype is not None else result


def _phase_progress(run: str, model: str, stage: str, task: str | None = None):
    def callback(step: int, total: int, anchor: str, loss: float) -> None:
        progress(
            "TRAIN_PROGRESS", run=run, model=model, stage=stage, task=task,
            step=step, total=total, anchor=anchor, loss=loss,
        )
    return callback


def _warmup_steps(total_steps: int) -> int:
    return 0 if total_steps < 2 else max(1, total_steps // 10)


def fit_gru_pretrainer(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    device: torch.device,
    budget: NeuralBudget | None = None,
    recipe: NeuralRecipe | None = None,
) -> tuple[Pretrainer, TrainingStats]:
    if model_id not in {"s1", "s2"}:
        raise ValueError(f"Unknown GRU SSL model: {model_id}")
    budget = budget or EXPERIMENT.budgets[model_id]
    if budget.ssl_steps <= 0:
        encoder_dropout = recipe.encoder_dropout if recipe else 0.2
        head_dropout = recipe.head_dropout if recipe else 0.2
        model = S1MaskedPretrainer(encoder_dropout=encoder_dropout, head_dropout=head_dropout) if model_id == "s1" else S2MultiHorizonPretrainer(encoder_dropout=encoder_dropout, head_dropout=head_dropout)
        return model, _zero_stats()
    seed_everything(EXPERIMENT.root_seed, run, model_id, "ssl", "model_init")
    encoder_dropout = recipe.encoder_dropout if recipe else 0.2
    head_dropout = recipe.head_dropout if recipe else 0.2
    model: Pretrainer = S1MaskedPretrainer(encoder_dropout=encoder_dropout, head_dropout=head_dropout) if model_id == "s1" else S2MultiHorizonPretrainer(encoder_dropout=encoder_dropout, head_dropout=head_dropout)
    model.to(device).train()
    factories = dense_ssl_factories(
        stores, anchors, run=run, model_id=model_id, batch_size=budget.batch_size
    )
    stream = round_robin_batches(factories, getattr(stores, "anchor_tickets", None), synchronized_epochs=recipe.synchronized_epochs if recipe else False)
    opt = recipe.ssl if recipe else OptimizerRecipe(1e-3, 1e-4, _warmup_steps(budget.ssl_steps))
    control = StepOptimizer(model.parameters(), total_steps=budget.ssl_steps, learning_rate=opt.learning_rate, weight_decay=opt.weight_decay, warmup_steps=opt.warmup_steps, scheduler=opt.scheduler, device=device)
    progress("TRAIN_START", run=run, model=model_id, stage="ssl", steps=budget.ssl_steps)

    def train_step(batch: tuple[torch.Tensor, ...]) -> float:
        control.prepare()
        x = _to_device(batch[0], device, dtype=torch.float32)
        with _autocast(device):
            if model_id == "s1":
                assert isinstance(model, S1MaskedPretrainer)
                corrupted, mask = model.corrupt(x)
                loss = model.loss(model(corrupted), x, mask)
            else:
                assert isinstance(model, S2MultiHorizonPretrainer)
                future = {
                    horizon: _to_device(value, device, dtype=torch.float32)
                    for horizon, value in zip((7, 14, 30), batch[1:])
                }
                buy = {horizon: (value > 0).float() for horizon, value in future.items()}
                z = {horizon: torch.log1p(value) for horizon, value in future.items()}
                loss = model.loss(model(x), buy, z)
        control.backward(loss)
        control.finish()
        return float(loss.detach().float().cpu())

    stats = train_exact_steps(
        batches=stream, requested_steps=budget.ssl_steps, train_step=train_step,
        progress=_phase_progress(run, model_id, "ssl"),
    )
    control.assert_complete()
    progress("TRAIN_DONE", run=run, model=model_id, stage="ssl", **stats.__dict__)
    return model, stats


def fit_gru_base(
    pretrainer: Pretrainer,
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    device: torch.device,
    budget: NeuralBudget | None = None,
    recipe: NeuralRecipe | None = None,
) -> BaseFit:
    budget = budget or EXPERIMENT.budgets[model_id]
    seed_everything(EXPERIMENT.root_seed, run, model_id, "base", "heads_init")
    encoder = pretrainer.encoder
    base = TransitionBase(encoder, lambda x: encoder(x)[1], head_dropout=recipe.head_dropout if recipe else 0.2).to(device).train()
    factories = dense_base_factories(
        stores, anchors, run=run, model_id=model_id, batch_size=budget.batch_size
    )
    stream = round_robin_batches(factories, getattr(stores, "anchor_tickets", None), synchronized_epochs=recipe.synchronized_epochs if recipe else False)
    opt = recipe.base if recipe else OptimizerRecipe(5e-4, 1e-4, _warmup_steps(budget.base_steps))
    control = StepOptimizer(base.parameters(), total_steps=budget.base_steps, learning_rate=opt.learning_rate, weight_decay=opt.weight_decay, warmup_steps=opt.warmup_steps, scheduler=opt.scheduler, device=device)
    progress("TRAIN_START", run=run, model=model_id, stage="base", steps=budget.base_steps)

    def train_step(batch: tuple[torch.Tensor, ...]) -> float:
        control.prepare()
        x, z, active, buy = batch
        x = _to_device(x, device, dtype=torch.float32)
        z = _to_device(z, device, dtype=torch.float32)
        active = _to_device(active, device, dtype=torch.float32)
        buy = _to_device(buy, device, dtype=torch.float32)
        with _autocast(device):
            weights = recipe.loss_weights if recipe else None
            loss = transition_loss(base(x), z, active, buy, **({"factorized_weight": weights.factorized, "direct_amount_weight": weights.direct_amount, "conditional_amount_weight": weights.conditional_amount, "react_weight": weights.react, "churn_weight": weights.churn} if weights else {}))
        control.backward(loss)
        control.finish()
        return float(loss.detach().float().cpu())

    stats = train_exact_steps(
        batches=stream, requested_steps=budget.base_steps, train_step=train_step,
        progress=_phase_progress(run, model_id, "base"),
    )
    control.assert_complete()
    progress("TRAIN_DONE", run=run, model=model_id, stage="base", **stats.__dict__)
    return BaseFit(base, stats)


def _fit_dense_specialist_phase(
    model: Specialist,
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    task: str,
    phase: str,
    steps: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    optimizer_recipe: OptimizerRecipe | None = None,
    recipe: NeuralRecipe | None = None,
) -> TrainingStats:
    factories = dense_specialist_factories(
        stores, anchors, run=run, model_id=model_id, task=task, phase=phase,
        batch_size=batch_size,
    )
    stream = round_robin_batches(factories, getattr(stores, "anchor_tickets", None), synchronized_epochs=recipe.synchronized_epochs if recipe else False)
    opt = optimizer_recipe or OptimizerRecipe(learning_rate, 1e-4, 0)
    control = StepOptimizer(model.parameters(), total_steps=steps, learning_rate=opt.learning_rate, weight_decay=opt.weight_decay, warmup_steps=opt.warmup_steps, scheduler=opt.scheduler, device=device)
    model.train()

    def train_step(batch: tuple[torch.Tensor, ...]) -> float:
        control.prepare()
        x, target = batch
        x = _to_device(x, device, dtype=torch.float32)
        target = _to_device(target, device, dtype=torch.float32)
        with _autocast(device):
            prediction = model(x)
            loss = (
                F.mse_loss(prediction, target)
                if task == "amount"
                else F.binary_cross_entropy_with_logits(prediction, target)
            )
        control.backward(loss)
        control.finish()
        return float(loss.detach().float().cpu())

    stats = train_exact_steps(
        batches=stream, requested_steps=steps, train_step=train_step,
        progress=_phase_progress(run, model_id, phase, task),
    )
    control.assert_complete()
    return stats


def fit_gru_specialist(
    base: TransitionBase,
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    model_id: str,
    task: str,
    device: torch.device,
    budget: NeuralBudget | None = None,
    recipe: NeuralRecipe | None = None,
    specialist_recipes: dict[str, tuple[OptimizerRecipe, OptimizerRecipe]] | None = None,
) -> SpecialistFit:
    budget = budget or EXPERIMENT.budgets[model_id]
    seed_everything(EXPERIMENT.root_seed, run, model_id, "specialist", task, "model_init")
    model = Specialist(base.encoder, task, model_id, head_dropout=recipe.head_dropout if recipe else 0.2).to(device)
    model.freeze_phase_h()
    progress("TRAIN_START", run=run, model=model_id, stage="H", task=task, steps=budget.specialist_head_steps)
    task_recipes = specialist_recipes.get(task) if specialist_recipes else None
    h_stats = _fit_dense_specialist_phase(
        model, stores, anchors, run=run, model_id=model_id, task=task, phase="H",
        steps=budget.specialist_head_steps, learning_rate=1e-3, optimizer_recipe=task_recipes[0] if task_recipes else (recipe.specialist_head if recipe else None),
        batch_size=budget.batch_size, device=device, recipe=recipe,
    )
    model.unfreeze_phase_f()
    progress("TRAIN_START", run=run, model=model_id, stage="F", task=task, steps=budget.specialist_finetune_steps)
    f_stats = _fit_dense_specialist_phase(
        model, stores, anchors, run=run, model_id=model_id, task=task, phase="F",
        steps=budget.specialist_finetune_steps, learning_rate=1e-4, optimizer_recipe=task_recipes[1] if task_recipes else (recipe.specialist_finetune if recipe else None),
        batch_size=budget.batch_size, device=device, recipe=recipe,
    )
    progress("TRAIN_DONE", run=run, model=model_id, stage="specialist", task=task)
    return SpecialistFit(model, {"H": h_stats, "F": f_stats})


def _event_inputs(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    content, time_features, ranks, padding_mask, empty = batch[:5]
    return (
        _to_device(content, device, dtype=torch.float32),
        _to_device(time_features, device, dtype=torch.float32),
        _to_device(ranks, device, dtype=torch.long),
        _to_device(padding_mask, device, dtype=torch.bool),
        _to_device(empty, device, dtype=torch.bool),
    )


def fit_ett_base(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    device: torch.device,
    budget: NeuralBudget | None = None,
    micro_batch_size: int = 128,
    accumulation_steps: int = 4,
    recipe: NeuralRecipe | None = None,
) -> BaseFit:
    budget = budget or EXPERIMENT.budgets["ett"]
    seed_everything(EXPERIMENT.root_seed, run, "ett", "base", "model_init")
    model = EventTimeTransformer(transformer_dropout=recipe.transformer_dropout if recipe else 0.1, head_dropout=recipe.head_dropout if recipe else 0.2).to(device).train()
    factories = event_base_factories(
        stores, anchors, run=run, batch_size=micro_batch_size
    )
    stream = round_robin_batches(factories, getattr(stores, "anchor_tickets", None), synchronized_epochs=recipe.synchronized_epochs if recipe else False)
    opt = recipe.base if recipe else OptimizerRecipe(3e-4, 1e-4, _warmup_steps(budget.base_steps))
    control = StepOptimizer(model.parameters(), total_steps=budget.base_steps, learning_rate=opt.learning_rate, weight_decay=opt.weight_decay, warmup_steps=opt.warmup_steps, scheduler=opt.scheduler, device=device)
    pending = 0
    progress("TRAIN_START", run=run, model="ett", stage="base", steps=budget.base_steps, accumulation=accumulation_steps)

    def micro_step(batch: tuple[torch.Tensor, ...], divisor: int) -> float:
        nonlocal pending
        if pending == 0:
            control.prepare()
        inputs = _event_inputs(batch, device)
        z = _to_device(batch[5], device, dtype=torch.float32)
        active = _to_device(batch[6], device, dtype=torch.float32)
        buy = _to_device(batch[7], device, dtype=torch.float32)
        with _autocast(device):
            weights = recipe.loss_weights if recipe else None
            unscaled_loss = transition_loss(model(*inputs), z, active, buy, **({"factorized_weight": weights.factorized, "direct_amount_weight": weights.direct_amount, "conditional_amount_weight": weights.conditional_amount, "react_weight": weights.react, "churn_weight": weights.churn} if weights else {}))
            loss = unscaled_loss / divisor
        control.backward(loss)
        pending += 1
        return float(unscaled_loss.detach().float().cpu())

    def optimizer_step() -> None:
        nonlocal pending
        if pending != accumulation_steps:
            raise RuntimeError("Incomplete ETT gradient accumulation group")
        control.finish()
        pending = 0

    stats = train_exact_accumulated_steps(
        batches=stream, requested_steps=budget.base_steps,
        accumulation_steps=accumulation_steps, micro_step=micro_step,
        optimizer_step=optimizer_step, progress=_phase_progress(run, "ett", "base"),
    )
    control.assert_complete()
    progress("TRAIN_DONE", run=run, model="ett", stage="base", **stats.__dict__)
    return BaseFit(model, stats)


def _fit_ett_specialist_phase(
    model: Specialist,
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    task: str,
    phase: str,
    steps: int,
    learning_rate: float,
    device: torch.device,
    micro_batch_size: int,
    accumulation_steps: int,
    optimizer_recipe: OptimizerRecipe | None = None,
    recipe: NeuralRecipe | None = None,
) -> TrainingStats:
    factories = event_specialist_factories(
        stores, anchors, run=run, task=task, phase=phase, batch_size=micro_batch_size
    )
    stream = round_robin_batches(factories, getattr(stores, "anchor_tickets", None), synchronized_epochs=recipe.synchronized_epochs if recipe else False)
    opt = optimizer_recipe or OptimizerRecipe(learning_rate, 1e-4, 0)
    control = StepOptimizer(model.parameters(), total_steps=steps, learning_rate=opt.learning_rate, weight_decay=opt.weight_decay, warmup_steps=opt.warmup_steps, scheduler=opt.scheduler, device=device)
    pending = 0
    model.train()

    def micro_step(batch: tuple[torch.Tensor, ...], divisor: int) -> float:
        nonlocal pending
        if pending == 0:
            control.prepare()
        inputs = _event_inputs(batch, device)
        target = _to_device(batch[5], device, dtype=torch.float32)
        with _autocast(device):
            prediction = model(*inputs)
            unscaled_loss = (
                F.mse_loss(prediction, target)
                if task == "amount"
                else F.binary_cross_entropy_with_logits(prediction, target)
            )
            loss = unscaled_loss / divisor
        control.backward(loss)
        pending += 1
        return float(unscaled_loss.detach().float().cpu())

    def optimizer_step() -> None:
        nonlocal pending
        if pending != accumulation_steps:
            raise RuntimeError("Incomplete ETT specialist accumulation group")
        control.finish()
        pending = 0

    stats = train_exact_accumulated_steps(
        batches=stream, requested_steps=steps, accumulation_steps=accumulation_steps,
        micro_step=micro_step, optimizer_step=optimizer_step,
        progress=_phase_progress(run, "ett", phase, task),
    )
    control.assert_complete()
    return stats


def fit_ett_specialist(
    base: EventTimeTransformer,
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    *,
    run: str,
    task: str,
    device: torch.device,
    budget: NeuralBudget | None = None,
    micro_batch_size: int = 128,
    accumulation_steps: int = 4,
    recipe: NeuralRecipe | None = None,
    specialist_recipes: dict[str, tuple[OptimizerRecipe, OptimizerRecipe]] | None = None,
) -> SpecialistFit:
    budget = budget or EXPERIMENT.budgets["ett"]
    seed_everything(EXPERIMENT.root_seed, run, "ett", "specialist", task, "model_init")
    model = Specialist(base, task, "ett", head_dropout=recipe.head_dropout if recipe else 0.2).to(device)
    model.freeze_phase_h()
    task_recipes = specialist_recipes.get(task) if specialist_recipes else None
    h_stats = _fit_ett_specialist_phase(
        model, stores, anchors, run=run, task=task, phase="H",
        steps=budget.specialist_head_steps, learning_rate=1e-3, device=device, optimizer_recipe=task_recipes[0] if task_recipes else (recipe.specialist_head if recipe else None),
        micro_batch_size=micro_batch_size, accumulation_steps=accumulation_steps, recipe=recipe,
    )
    model.unfreeze_phase_f()
    f_stats = _fit_ett_specialist_phase(
        model, stores, anchors, run=run, task=task, phase="F",
        steps=budget.specialist_finetune_steps, learning_rate=1e-4, device=device, optimizer_recipe=task_recipes[1] if task_recipes else (recipe.specialist_finetune if recipe else None),
        micro_batch_size=micro_batch_size, accumulation_steps=accumulation_steps, recipe=recipe,
    )
    progress("TRAIN_DONE", run=run, model="ett", stage="specialist", task=task)
    return SpecialistFit(model, {"H": h_stats, "F": f_stats})
