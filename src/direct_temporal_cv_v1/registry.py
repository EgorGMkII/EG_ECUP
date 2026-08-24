"""Direct model registry.  Implementations deliberately remain thin wrappers."""

from __future__ import annotations

from .base import DirectModelAdapter, ModelRequirements
from .adapters.catboost_direct import DirectCatBoostAdapter
from .adapters.direct_ett import DirectETTAdapter
from .adapters.direct_tcn import DirectTCNAdapter


class _UnimplementedAdapter(DirectModelAdapter):
    """Temporary skeleton; concrete adapters replace its methods."""

    def __init__(self, model_id: str, requirements: ModelRequirements) -> None:
        self.model_id = model_id
        self.requirements = requirements

    def validate_config(self, raw):  # type: ignore[no-untyped-def]
        raise NotImplementedError(f"{self.model_id} adapter has not been filled in")

    def fit_predict_fold(self, context, config):  # type: ignore[no-untyped-def]
        raise NotImplementedError(f"{self.model_id} adapter has not been filled in")


MODEL_REGISTRY = {
    "catboost_direct": DirectCatBoostAdapter,
    # Neural adapters are intentionally still skeletons. Keeping their IDs in
    # the registry gives config validation a stable extension point without
    # accidentally launching an incomplete model.
    "ett_direct": DirectETTAdapter,
    "tcn_direct": DirectTCNAdapter,
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
