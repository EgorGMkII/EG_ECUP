"""Direct model registry.  Implementations deliberately remain thin wrappers."""

from __future__ import annotations

from .base import DirectModelAdapter, ModelRequirements
from .adapters.catboost_direct import DirectCatBoostAdapter
from .adapters.direct_ett import DirectETTAdapter
from .adapters.direct_tcn import DirectTCNAdapter
from .adapters.catboost_cohort_specialist import CatBoostCohortSpecialistAdapter
from .adapters.ett_classifier import ETTClassifierAdapter
from .adapters.sequential_churn_classifier import SequentialChurnClassifierAdapter
from .adapters.hybrid_cohort_specialist import HybridCohortSpecialistAdapter
from .adapters.lightgbm_direct import DirectLightGBMAdapter
from .adapters.catboost_tweedie import DirectCatBoostTweedieAdapter
from .adapters.lightgbm_tweedie import DirectLightGBMTweedieAdapter
from .adapters.frequency_specialist import DirectFrequencySpecialistAdapter
from .adapters.delta_regressor import DirectDeltaRegressorAdapter
from .adapters.two_tower_adapter import DirectTwoTowerAdapter


MODEL_REGISTRY = {
    "catboost_direct": DirectCatBoostAdapter,
    "catboost_cohort_specialist": CatBoostCohortSpecialistAdapter,
    "lightgbm_direct": DirectLightGBMAdapter,
    "catboost_tweedie": DirectCatBoostTweedieAdapter,
    "lightgbm_tweedie": DirectLightGBMTweedieAdapter,
    "direct_frequency_specialist": DirectFrequencySpecialistAdapter,
    "direct_delta_regressor": DirectDeltaRegressorAdapter,
    "two_tower_direct": DirectTwoTowerAdapter,
    "ett_direct": DirectETTAdapter,
    "tcn_direct": DirectTCNAdapter,
    "ett_classifier": ETTClassifierAdapter,
    "sequential_churn_classifier": SequentialChurnClassifierAdapter,
    "hybrid_cohort_specialist": HybridCohortSpecialistAdapter,
}


def build_adapters(model_ids: tuple[str, ...]) -> list[DirectModelAdapter]:
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("enabled direct model IDs must be unique")
    try:
        adapters = [MODEL_REGISTRY[model_id]() for model_id in model_ids]
    except KeyError as error:
        raise ValueError(f"Unknown direct model ID: {error.args[0]}") from error
    return adapters


def collect_requirements(adapters: list[DirectModelAdapter]) -> ModelRequirements:
    return ModelRequirements(
        tabular_features=any(adapter.requirements.tabular_features for adapter in adapters),
        daily_tensor=any(adapter.requirements.daily_tensor for adapter in adapters),
        event_sequences=any(adapter.requirements.event_sequences for adapter in adapters),
        btyd_features=any(adapter.requirements.btyd_features for adapter in adapters),
    )
