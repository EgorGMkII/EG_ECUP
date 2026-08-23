from __future__ import annotations

import numpy as np

from src.ssl_temporal_stack_v1.diagnostics import build_validation_report


def test_validation_report_contains_all_transitions_and_provenance() -> None:
    active = np.array([0, 0, 1, 1], dtype=np.int8)
    buy = np.array([0, 1, 0, 1], dtype=np.int8)
    components = {
        "p_react": np.array([0.1, 0.8, 0.5, 0.5]),
        "p_churn": np.array([0.5, 0.5, 0.8, 0.1]),
        "p_buy": np.array([0.1, 0.8, 0.2, 0.9]),
        "conditional_z": np.ones(4),
        "prediction_z": np.array([0.0, 0.8, 0.2, 0.9]),
    }
    report = build_validation_report(
        target_z=np.array([0.0, 1.0, 0.0, 1.0]), was_active=active, will_buy=buy,
        components=components, validation_anchor="2026-01-14", job_id="job",
        commit_sha="commit", config_sha256="config", bank_sha256="bank", meta_sha256="meta",
    )
    assert set(report["transitions"]) == {"00", "01", "10", "11"}
    assert all(item["rows"] == 1 for item in report["transitions"].values())
    assert report["provenance"]["job_id"] == "job"
