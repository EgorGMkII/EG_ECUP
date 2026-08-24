"""Adapters that delegate to proven SSL V1 implementations."""

from __future__ import annotations

import gc
from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
import polars as pl

from src.ssl_temporal_stack_v1.adapters import fit_predict_catboost, fit_predict_ett, fit_predict_gru, resolved_catboost_params
from src.ssl_temporal_stack_v1.contract import NeuralBudget
from src.ssl_temporal_stack_v1.recipes import LossWeights, NeuralRecipe, OptimizerRecipe
from src.ssl_temporal_stack_v1.runtime import derive_seed, progress

from .base import FirstLevelAdapter, ModelConfig, ModelResult, PredictionSpec, RunContext
from .candidate_adapters import fit_predict_residual_mlp, fit_predict_tcn
from .candidates.btyd import AuditedBTYDClassifierProvider, BTYDRecipe
from .direct_catboost import fit_predict_direct_catboost


class _SSLConfigShim:
    def __init__(self, models: Mapping[str, Any]) -> None:
        self.raw = {"models": models}


class _Adapter(FirstLevelAdapter):
    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{self.model_id} config must be a mapping")
        return ModelConfig(self.model_id, dict(raw))


def _stage(raw: Mapping[str, Any], location: str, *, allow_zero: bool = False) -> None:
    allowed = {"steps", "epochs", "learning_rate", "scheduler", "warmup_steps", "warmup_fraction", "weight_decay", "checkpoint_every"}
    if set(raw) - allowed or not {"steps", "learning_rate"} <= set(raw):
        if not ({"epochs", "learning_rate"} <= set(raw) and "steps" not in raw):
            raise ValueError(f"Invalid stage recipe at {location}")
    if "steps" in raw and "epochs" in raw:
        raise ValueError(f"Specify only steps or epochs at {location}")
    budget = int(raw.get("steps", raw.get("epochs", 0)))
    if budget < (0 if allow_zero else 1) or float(raw["learning_rate"]) < 0:
        raise ValueError(f"Invalid stage values at {location}")
    if raw.get("scheduler", "cosine") not in {"constant", "linear", "cosine"}:
        raise ValueError(f"Unsupported scheduler at {location}")
    if "warmup_steps" in raw and "warmup_fraction" in raw:
        raise ValueError(f"Specify only warmup_steps or warmup_fraction at {location}")
    if not 0 <= int(raw.get("warmup_steps", 0)) < max(1, budget):
        raise ValueError(f"Invalid warmup at {location}")
    if not 0.0 <= float(raw.get("warmup_fraction", 0.0)) < 1.0:
        raise ValueError(f"Invalid warmup fraction at {location}")


class CatBoostAdapter(_Adapter):
    model_id = "catboost"
    required_stores = frozenset({"frames"})
    prediction_spec = PredictionSpec("catboost", "cb_react_logit", "cb_churn_logit", "cb_amount_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        required = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "task_type", "devices", "boosting_type", "grow_policy", "bootstrap_type", "bagging_temperature", "random_strength", "border_count", "nan_mode"}
        if set(raw) - (required | {"btyd"}) or not required <= set(raw):
            raise ValueError("CatBoost config has missing or unknown fields")
        if raw["task_type"] != "GPU":
            raise ValueError("Reference CatBoost requires task_type=GPU")
        if "btyd" in raw:
            if not isinstance(raw["btyd"], Mapping) or set(raw["btyd"]) - {"enabled", "penalizer_coef", "horizon_days", "max_fit_users"}:
                raise ValueError("Invalid CatBoost BTYD config")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        if not config.values.get("btyd", {}).get("enabled", False):
            result = fit_predict_catboost(context.stores, context.train_anchors, context.holdout_anchor, run=context.run_name, config=_SSLConfigShim({"catboost": dict(config.values)}))
            return ModelResult(result.model_id, result.predictions, result.training_report)
        result = self._fit_with_btyd(context, config)
        return ModelResult(result.model_id, result.predictions, result.training_report)

    @staticmethod
    def _fit_with_btyd(context: RunContext, config: ModelConfig) -> ModelResult:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool, __version__ as catboost_version

        raw = dict(config.values)
        btyd = raw.pop("btyd")
        if context.raw_events is None:
            raise RuntimeError("BTYD requires raw causal event history in RunContext")
        provider = AuditedBTYDClassifierProvider(
            BTYDRecipe(
                penalizer_coef=float(btyd.get("penalizer_coef", 0.001)),
                horizon_days=int(btyd.get("horizon_days", 30)),
                max_fit_users=int(btyd.get("max_fit_users", 50_000)),
            ),
            root_seed=derive_seed(context.root_seed, context.run_name, "btyd_fit"),
        )
        train_frames = [context.stores.frames.get(anchor) for anchor in context.train_anchors]
        btyd_tables = provider.fit_transform_anchors(context.raw_events, context.users, context.train_anchors, context.holdout_anchor)
        augmented_train = [frame.join(btyd_tables[anchor], on="user_id", how="left") for anchor, frame in zip(context.train_anchors, train_frames, strict=True)]
        holdout = context.stores.frames.get(context.holdout_anchor)
        augmented_holdout = holdout.join(btyd_tables[context.holdout_anchor], on="user_id", how="left")
        base_features = tuple(context.stores.frames.feature_names)
        classifier_features = (*base_features, *provider.feature_names)
        forbidden = {name for name in classifier_features if name in {"will_buy", "will_buy_30d", "future_gmv_30d", "z_target", "target"}}
        if forbidden:
            raise RuntimeError(f"Target leakage in CatBoost features: {sorted(forbidden)}")
        pooled = pl.concat(augmented_train)
        predictions: dict[str, np.ndarray] = {}
        tasks: dict[str, Any] = {}
        definitions = (
            ("react", pooled["was_active"].to_numpy() == 0, pooled["will_buy"].to_numpy(), CatBoostClassifier, "Logloss"),
            ("churn", pooled["was_active"].to_numpy() == 1, 1 - pooled["will_buy"].to_numpy(), CatBoostClassifier, "Logloss"),
            ("amount", pooled["future_gmv_30d"].to_numpy() > 0, pooled["z_target"].to_numpy(), CatBoostRegressor, "RMSE"),
        )
        for task, mask, full_target, model_class, loss_function in definitions:
            started = time.perf_counter()
            subset, target = pooled.filter(pl.Series(mask)), np.asarray(full_target)[mask]
            params = resolved_catboost_params(_SSLConfigShim({"catboost": raw}), seed=derive_seed(context.root_seed, context.run_name, "catboost_btyd", task))
            params["loss_function"] = loss_function
            progress("CATBOOST_START", run=context.run_name, task=task, rows=subset.height, btyd=True, params=params)
            model = model_class(**params)
            task_features = classifier_features if task in {"react", "churn"} else base_features
            model.fit(Pool(subset.select(task_features).to_pandas(), label=target))
            holdout_pool = Pool(augmented_holdout.select(task_features).to_pandas())
            output = model.predict(holdout_pool, prediction_type="RawFormulaVal") if task != "amount" else np.clip(model.predict(holdout_pool), 0.0, None)
            column = f"cb_{task}_logit" if task != "amount" else "cb_amount_z"
            predictions[column] = np.asarray(output, dtype=np.float64)
            tasks[task] = {"rows": subset.height, "elapsed_seconds": time.perf_counter() - started, "tree_count": int(model.tree_count_), "resolved_parameters": model.get_all_params()}
            progress("CATBOOST_DONE", run=context.run_name, task=task, btyd=True, **tasks[task])
            del subset, target, model
            gc.collect()
        return ModelResult("catboost", predictions, {"catboost_version": catboost_version, "tasks": tasks, "btyd": {"feature_set_id": provider.feature_set_id, "classifier_features": list(provider.feature_names), "amount_features": [], "fit_anchors": list(context.train_anchors), "fit_rows": provider.fit_rows, "history": "full causal history through each anchor"}})


class DirectCatBoostAdapter(_Adapter):
    """Unconditional log-GMV model blended only after the hurdle branch is complete."""

    model_id = "catboost_direct"
    required_stores = frozenset({"frames"})
    prediction_spec = PredictionSpec("catboost_direct", None, None, None, "cb_direct_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        required = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "task_type", "devices", "training_anchors"}
        if set(raw) != required:
            raise ValueError(f"catboost_direct fields must be exactly {sorted(required)}")
        if raw["task_type"] != "GPU" or raw["training_anchors"] != "holdout_minus_30d":
            raise ValueError("Direct CatBoost requires GPU and holdout-minus-30d training")
        if int(raw["iterations"]) < 1 or int(raw["depth"]) < 1 or float(raw["learning_rate"]) <= 0:
            raise ValueError("Invalid direct CatBoost parameters")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        prediction, report = fit_predict_direct_catboost(context, config.values)
        return ModelResult(self.model_id, {"cb_direct_z": prediction}, report)


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


class TCNAdapter(_Adapter):
    model_id = "tcn"
    required_stores = frozenset({"frames", "daily"})
    prediction_spec = PredictionSpec("tcn", "tcn_react_logit", "tcn_churn_logit", None)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"batch_size", "channels", "dropout", "head_dropout", "history_days", "ssl", "base", "specialists", "loss_weights"}
        if set(raw) - allowed or not {"batch_size", "base", "specialists"} <= set(raw):
            raise ValueError("Invalid TCN config")
        if raw.get("ssl", "disabled") != "disabled":
            raise ValueError("TCN V1 supports only ssl: disabled")
        history_days = int(raw.get("history_days", 180))
        if history_days < 30 or history_days > 365:
            raise ValueError("TCN history_days must be in [30, 365]")
        _stage(raw["base"], "tcn.base")
        if set(raw["specialists"]) != {"react", "churn"}:
            raise ValueError("TCN must configure React and Churn only")
        for task in ("react", "churn"):
            if set(raw["specialists"][task]) != {"H", "F"}:
                raise ValueError(f"Invalid TCN specialist phases for {task}")
            _stage(raw["specialists"][task]["H"], f"tcn.{task}.H")
            _stage(raw["specialists"][task]["F"], f"tcn.{task}.F")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        return fit_predict_tcn(context, dict(config.values))


class ResidualMLPAdapter(_Adapter):
    model_id = "residual_mlp"
    required_stores = frozenset({"frames"})
    prediction_spec = PredictionSpec("residual_mlp", "mlp_react_logit", "mlp_churn_logit", "mlp_amount_z")

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"batch_size", "hidden", "blocks", "dropout", "head_dropout", "ssl", "base", "specialists", "loss_weights"}
        if set(raw) - allowed or not {"batch_size", "base", "specialists"} <= set(raw):
            raise ValueError("Invalid Residual MLP config")
        if raw.get("ssl", "disabled") != "disabled":
            raise ValueError("Residual MLP V1 supports only ssl: disabled")
        _stage(raw["base"], "residual_mlp.base")
        if set(raw["specialists"]) != {"react", "churn", "amount"}:
            raise ValueError("Residual MLP must configure all specialists")
        for task in ("react", "churn", "amount"):
            if set(raw["specialists"][task]) != {"H", "F"}:
                raise ValueError(f"Invalid Residual MLP specialist phases for {task}")
            _stage(raw["specialists"][task]["H"], f"residual_mlp.{task}.H")
            _stage(raw["specialists"][task]["F"], f"residual_mlp.{task}.F")
        return super().validate_config(raw)

    def fit_predict(self, context: RunContext, config: ModelConfig) -> ModelResult:
        return fit_predict_residual_mlp(context, dict(config.values))


MODEL_REGISTRY = {"catboost": CatBoostAdapter, "catboost_direct": DirectCatBoostAdapter, "s1": lambda: GRUAdapter("s1"), "s2": lambda: GRUAdapter("s2"), "ett": ETTAdapter, "tcn": TCNAdapter, "residual_mlp": ResidualMLPAdapter}


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
        synchronized_epochs=any(
            isinstance(stage, Mapping) and "epoch_resolution" in stage
            for stage in [values.get("ssl"), values.get("base"), *(phase for task in values["specialists"].values() for phase in task.values())]
        ),
    )
    return recipe, specialists


def build_adapters(model_ids: tuple[str, ...]) -> list[FirstLevelAdapter]:
    try:
        return [MODEL_REGISTRY[model_id]() for model_id in model_ids]
    except KeyError as error:
        raise ValueError(f"Unknown model ID: {error.args[0]}") from error


def collect_required_stores(adapters: list[FirstLevelAdapter]) -> frozenset[str]:
    return frozenset().union(*(adapter.required_stores for adapter in adapters))
