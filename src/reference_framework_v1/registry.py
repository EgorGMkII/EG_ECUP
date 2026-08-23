"""Adapters that delegate to proven SSL V1 implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.ssl_temporal_stack_v1.adapters import fit_predict_catboost, fit_predict_ett, fit_predict_gru
from src.ssl_temporal_stack_v1.contract import NeuralBudget
from src.ssl_temporal_stack_v1.recipes import LossWeights, NeuralRecipe, OptimizerRecipe

from .base import FirstLevelAdapter, ModelConfig, ModelResult, PredictionSpec, RunContext


class _SSLConfigShim:
    def __init__(self, models: Mapping[str, Any]) -> None:
        self.raw = {"models": models}


class _Adapter(FirstLevelAdapter):
    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{self.model_id} config must be a mapping")
        return ModelConfig(self.model_id, dict(raw))


def _stage(raw: Mapping[str, Any], location: str, *, allow_zero: bool = False) -> None:
    allowed = {"steps", "learning_rate", "scheduler", "warmup_steps", "weight_decay", "checkpoint_every"}
    if set(raw) - allowed or not {"steps", "learning_rate"} <= set(raw):
        raise ValueError(f"Invalid stage recipe at {location}")
    if int(raw["steps"]) < (0 if allow_zero else 1) or float(raw["learning_rate"]) < 0:
        raise ValueError(f"Invalid stage values at {location}")
    if raw.get("scheduler", "cosine") not in {"constant", "linear", "cosine"}:
        raise ValueError(f"Unsupported scheduler at {location}")
    if not 0 <= int(raw.get("warmup_steps", 0)) < max(1, int(raw["steps"])):
        raise ValueError(f"Invalid warmup at {location}")


class CatBoostAdapter(_Adapter):
    model_id = "catboost"
    required_stores = frozenset({"frames"})
    prediction_spec = PredictionSpec("catboost", "cb_react_logit", "cb_churn_logit", "cb_amount_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        required = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "task_type", "devices", "boosting_type", "grow_policy", "bootstrap_type", "bagging_temperature", "random_strength", "border_count", "nan_mode"}
        if set(raw) != required:
            raise ValueError("CatBoost config has missing or unknown fields")
        if raw["task_type"] != "GPU":
            raise ValueError("Reference CatBoost requires task_type=GPU")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        result = fit_predict_catboost(context.stores, context.train_anchors, context.holdout_anchor, run=context.run_name, config=_SSLConfigShim({"catboost": dict(config.values)}))
        return ModelResult(result.model_id, result.predictions, result.training_report)


class GRUAdapter(_Adapter):
    required_stores = frozenset({"frames", "daily", "horizons"})

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.prediction_spec = PredictionSpec(model_id, f"{model_id}_react_logit", f"{model_id}_churn_logit", f"{model_id}_amount_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        if set(raw) - {"batch_size", "encoder_dropout", "head_dropout", "ssl", "base", "specialists", "loss_weights"}:
            raise ValueError(f"Unknown {self.model_id} config field")
        if not {"batch_size", "ssl", "base", "specialists"} <= set(raw):
            raise ValueError(f"Missing {self.model_id} config field")
        _stage(raw["ssl"], f"{self.model_id}.ssl", allow_zero=True)
        _stage(raw["base"], f"{self.model_id}.base")
        if set(raw["specialists"]) != {"react", "churn", "amount"}:
            raise ValueError("Each specialist task must be configured")
        for task, phases in raw["specialists"].items():
            if set(phases) != {"H", "F"}:
                raise ValueError(f"Invalid specialist phases for {task}")
            _stage(phases["H"], f"{self.model_id}.{task}.H")
            _stage(phases["F"], f"{self.model_id}.{task}.F")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        values = config.values
        recipe, specialist_recipes = _neural_recipes(values)
        budget = NeuralBudget(int(values["ssl"]["steps"]), int(values["base"]["steps"]), int(values["specialists"]["react"]["H"]["steps"]), int(values["specialists"]["react"]["F"]["steps"]), int(values["batch_size"]))
        result = fit_predict_gru(context.stores, context.train_anchors, context.holdout_anchor, run=context.run_name, model_id=self.model_id, device=context.device, checkpoint_root=context.output_dir / "_work" / "checkpoints", budget=budget, recipe=recipe, specialist_recipes=specialist_recipes)
        return ModelResult(result.model_id, result.predictions, {**result.training_report, "resolved_recipe": dict(values)})


class ETTAdapter(_Adapter):
    model_id = "ett"
    required_stores = frozenset({"frames", "events"})
    prediction_spec = PredictionSpec("ett", "ett_react_logit", "ett_churn_logit", "ett_amount_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        if set(raw) - {"effective_batch_size", "micro_batch_size", "accumulation_steps", "transformer_dropout", "head_dropout", "base", "specialists", "loss_weights"}:
            raise ValueError("Unknown ETT config field")
        if not {"effective_batch_size", "micro_batch_size", "accumulation_steps", "base", "specialists"} <= set(raw):
            raise ValueError("Missing ETT config field")
        _stage(raw["base"], "ett.base")
        if set(raw["specialists"]) != {"react", "churn", "amount"}:
            raise ValueError("Each ETT specialist task must be configured")
        for task, phases in raw["specialists"].items():
            if set(phases) != {"H", "F"}:
                raise ValueError(f"Invalid ETT specialist phases for {task}")
            _stage(phases["H"], f"ett.{task}.H")
            _stage(phases["F"], f"ett.{task}.F")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        values = config.values
        recipe, specialist_recipes = _neural_recipes(values)
        budget = NeuralBudget(0, int(values["base"]["steps"]), int(values["specialists"]["react"]["H"]["steps"]), int(values["specialists"]["react"]["F"]["steps"]), int(values["effective_batch_size"]))
        result = fit_predict_ett(context.stores, context.train_anchors, context.holdout_anchor, run=context.run_name, device=context.device, checkpoint_root=context.output_dir / "_work" / "checkpoints", budget=budget, micro_batch_size=int(values["micro_batch_size"]), accumulation_steps=int(values["accumulation_steps"]), recipe=recipe, specialist_recipes=specialist_recipes)
        return ModelResult(result.model_id, result.predictions, {**result.training_report, "resolved_recipe": dict(values)})


MODEL_REGISTRY = {"catboost": CatBoostAdapter, "s1": lambda: GRUAdapter("s1"), "s2": lambda: GRUAdapter("s2"), "ett": ETTAdapter}


def _optimizer(raw: Mapping[str, Any]) -> OptimizerRecipe:
    return OptimizerRecipe(float(raw["learning_rate"]), float(raw.get("weight_decay", 1e-4)), int(raw.get("warmup_steps", 0)), str(raw.get("scheduler", "cosine")), int(raw.get("steps", 0)))


def _neural_recipes(values: Mapping[str, Any]) -> tuple[NeuralRecipe, dict[str, tuple[OptimizerRecipe, OptimizerRecipe]]]:
    specialists = {task: (_optimizer(values["specialists"][task]["H"]), _optimizer(values["specialists"][task]["F"])) for task in ("react", "churn", "amount")}
    loss = values.get("loss_weights", {})
    recipe = NeuralRecipe(
        ssl=_optimizer(values.get("ssl", {"learning_rate": 0.0, "steps": 0})),
        base=_optimizer(values["base"]), specialist_head=specialists["react"][0], specialist_finetune=specialists["react"][1],
        encoder_dropout=float(values.get("encoder_dropout", 0.2)), head_dropout=float(values.get("head_dropout", 0.2)), transformer_dropout=float(values.get("transformer_dropout", 0.1)),
        loss_weights=LossWeights(**loss),
    )
    return recipe, specialists


def build_adapters(model_ids: tuple[str, ...]) -> list[FirstLevelAdapter]:
    try:
        return [MODEL_REGISTRY[model_id]() for model_id in model_ids]
    except KeyError as error:
        raise ValueError(f"Unknown model ID: {error.args[0]}") from error


def collect_required_stores(adapters: list[FirstLevelAdapter]) -> frozenset[str]:
    return frozenset().union(*(adapter.required_stores for adapter in adapters))
