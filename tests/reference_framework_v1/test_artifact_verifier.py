from pathlib import Path

import polars as pl
import pytest

from scripts.verify_reference_experiment_artifacts import _validate_bank


def test_screen_bank_verifier_accepts_its_intentional_base_column_order() -> None:
    # The machine's global pytest temp root is intentionally inaccessible;
    # use a narrowly owned workspace file and always remove just that file.
    path = Path("tests/reference_framework_v1/.artifact_verifier_test.parquet")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite stale test file: {path}")
    # Individual screens retain the adapter bank order; full runs use a
    # canonical base-column order.  Both must be schema-checked, not guessed.
    try:
        pl.DataFrame(
            {
                "user_id": [1, 2],
                "was_active": [0, 1],
                "will_buy": [0, 1],
                "future_gmv_30d": [0.0, 1.0],
                "z_target": [0.0, 0.69314718056],
                "anchor": ["2026-01-14", "2026-01-14"],
                "ett_react_logit": [0.1, 0.2],
            }
        ).write_parquet(path)
        schema = {"react": ["ett_react_logit"], "churn": [], "amount": [], "direct": []}

        _validate_bank(path, rows=2, schema=schema, strict_column_order=False)
        with pytest.raises(ValueError, match="Unexpected bank schema"):
            _validate_bank(path, rows=2, schema=schema)
    finally:
        if path.exists():
            path.unlink()
