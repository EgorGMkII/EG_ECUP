import numpy as np

from src.reference_framework_v1.meta import apply_meta, fit_meta
from src.reference_framework_v1.predictions import PredictionSchema
from src.reference_framework_v1.selection import _schema_from_package


def test_named_meta_supports_dynamic_columns() -> None:
    rng = np.random.default_rng(7)
    schema = PredictionSchema(("a_react", "b_react"), ("a_churn",), ("a_amount", "b_amount"))
    bank = {"react": rng.normal(size=(32, 2)), "churn": rng.normal(size=(32, 1)), "amount": rng.normal(size=(32, 2)), "active": rng.integers(0, 2, 32), "target": rng.random(32)}
    package = fit_meta(bank, schema, root_seed=42, prediction_bank_sha256="bank", commit_sha="commit", config_sha256="config")
    result = apply_meta(package, bank, schema)
    assert result.shape == (32,)
    assert np.isfinite(result).all()
    assert (result >= 0).all()


def test_direct_prediction_is_late_blended_with_complete_hurdle_branch() -> None:
    rng = np.random.default_rng(11)
    rows = 128
    target = rng.uniform(0.0, 4.0, rows)
    schema = PredictionSchema(("react",), ("churn",), ("amount",), ("direct",))
    bank = {
        "react": np.zeros((rows, 1)),
        "churn": np.zeros((rows, 1)),
        "amount": np.ones((rows, 1)),
        "direct": target[:, None],
        "active": rng.integers(0, 2, rows),
        "target": target,
    }
    package = fit_meta(bank, schema, root_seed=42, prediction_bank_sha256="bank", commit_sha="commit", config_sha256="config")
    assert package["meta_schema_version"] == 3
    assert set(package["late_blend_weights"]) == {"hurdle", "direct"}
    assert package["late_blend_weights"]["direct"] > 0.99
    assert np.allclose(apply_meta(package, bank, schema), target, atol=1e-5)


def test_selection_recovers_composite_schema_from_frozen_meta_package() -> None:
    schema = _schema_from_package({"feature_order": {"react": ["r"], "churn": ["c"], "amount": ["a"], "direct": ["d"]}})
    assert schema == PredictionSchema(("r",), ("c",), ("a",), ("d",))
