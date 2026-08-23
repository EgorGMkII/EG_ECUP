from __future__ import annotations

import numpy as np

from src.ssl_temporal_stack_v1.meta import apply_meta, fit_meta


def test_meta_fit_and_frozen_apply_are_finite() -> None:
    rng = np.random.default_rng(7)
    rows = 120
    bank = {
        "react": rng.normal(size=(rows, 4)),
        "churn": rng.normal(size=(rows, 4)),
        "amount": np.abs(rng.normal(size=(rows, 4))),
        "active": rng.integers(0, 2, size=rows).astype(np.int8),
        "target": np.abs(rng.normal(size=rows)),
    }
    package = fit_meta(
        bank, prediction_bank_sha256="bank", code_commit_sha="commit", config_sha256="config"
    )
    prediction = apply_meta(package, bank)
    assert prediction.shape == (rows,)
    assert np.isfinite(prediction).all()
    assert len(package["attempts"]) == 9
    assert package["rmsle"] >= 0
