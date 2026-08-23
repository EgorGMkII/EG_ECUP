import numpy as np

from src.reference_framework_v1.meta import apply_meta, fit_meta
from src.reference_framework_v1.predictions import PredictionSchema


def test_named_meta_supports_dynamic_columns() -> None:
    rng = np.random.default_rng(7)
    schema = PredictionSchema(("a_react", "b_react"), ("a_churn",), ("a_amount", "b_amount"))
    bank = {"react": rng.normal(size=(32, 2)), "churn": rng.normal(size=(32, 1)), "amount": rng.normal(size=(32, 2)), "active": rng.integers(0, 2, 32), "target": rng.random(32)}
    package = fit_meta(bank, schema, root_seed=42, prediction_bank_sha256="bank", commit_sha="commit", config_sha256="config")
    result = apply_meta(package, bank, schema)
    assert result.shape == (32,)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
