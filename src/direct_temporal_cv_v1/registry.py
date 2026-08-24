"""Direct model registry.  Implementations deliberately remain thin wrappers."""

from __future__ import annotations

from .base import DirectModelAdapter, ModelRequirements


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
    "catboost_direct": lambda: _UnimplementedAdapter("catboost_direct", ModelRequirements(tabular_features=True)),
    "ett_direct": lambda: _UnimplementedAdapter("ett_direct", ModelRequirements(event_sequences=True)),
    "tcn_direct": lambda: _UnimplementedAdapter("tcn_direct", ModelRequirements(daily_tensor=True)),
}


def build_adapters(model_ids: tuple[str, ...]) -> list[DirectModelAdapter]:
    try:
        return [MODEL_REGISTRY[model_id]() for model_id in model_ids]
    except KeyError as error:
        raise ValueError(f"Unknown direct model ID: {error.args[0]}") from error


def collect_requirements(adapters: list[DirectModelAdapter]) -> ModelRequirements:
    return ModelRequirements(
        tabular_features=any(adapter.requirements.tabular_features for adapter in adapters),
        daily_tensor=any(adapter.requirements.daily_tensor for adapter in adapters),
        event_sequences=any(adapter.requirements.event_sequences for adapter in adapters),
        btyd_features=any(adapter.requirements.btyd_features for adapter in adapters),
    )
