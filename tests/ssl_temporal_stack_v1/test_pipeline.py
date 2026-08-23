from __future__ import annotations

import numpy as np
import pytest

from src.ssl_temporal_stack_v1.adapters import AdapterResult
from src.ssl_temporal_stack_v1.config import load_config
from src.ssl_temporal_stack_v1.pipeline import contract_check, merge_adapter_results


def test_merge_adapter_results_requires_complete_schema() -> None:
    rows = 3
    results = []
    for model_id, prefix in (("catboost", "cb"), ("s1", "s1"), ("s2", "s2"), ("ett", "ett")):
        results.append(AdapterResult(model_id, {
            f"{prefix}_react_logit": np.zeros(rows),
            f"{prefix}_churn_logit": np.zeros(rows),
            f"{prefix}_amount_z": np.zeros(rows),
        }, {"ok": True}))
    predictions, reports = merge_adapter_results(results, expected_rows=rows)
    assert len(predictions) == 12
    assert set(reports) == {"catboost", "s1", "s2", "ett"}


def test_merge_adapter_results_rejects_incomplete_stack() -> None:
    with pytest.raises(ValueError, match="missing"):
        merge_adapter_results([
            AdapterResult("catboost", {
                "cb_react_logit": np.zeros(2), "cb_churn_logit": np.zeros(2),
                "cb_amount_z": np.zeros(2),
            }, {})
        ], expected_rows=2)


def test_contract_check_is_explicitly_not_training_smoke() -> None:
    result = contract_check(load_config())
    assert result["check_kind"] == "CONTRACT_ONLY_NO_TRAINING"
    assert len(result["prediction_columns"]) == 12
