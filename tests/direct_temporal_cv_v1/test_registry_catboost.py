from src.direct_temporal_cv_v1.registry import build_adapters, collect_requirements


def test_catboost_adapter_is_registered_and_requires_tabular() -> None:
    adapters = build_adapters(("catboost_direct",))
    assert [adapter.model_id for adapter in adapters] == ["catboost_direct"]
    requirements = collect_requirements(adapters)
    assert requirements.tabular_features is True
    assert requirements.daily_tensor is False
    assert requirements.event_sequences is False


def test_registry_rejects_duplicate_models() -> None:
    try:
        build_adapters(("catboost_direct", "catboost_direct"))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate model IDs must fail")
