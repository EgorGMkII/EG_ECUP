import importlib.util
from pathlib import Path



MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_recipe_attempt.py"
SPEC = importlib.util.spec_from_file_location("record_recipe_attempt", MODULE_PATH)
attempt = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(attempt)


def test_routed_paths_isolate_only_legacy_outputs() -> None:
    output_dir = Path("artifacts") / "isolated_test_run"
    route = attempt.routed_path_factory(output_dir)
    assert route("submission_specialized_hurdle_joint_rmsle.csv") == output_dir / "candidate_submission.csv"
    assert route("artifacts/specialized_hurdle/feature_store") == output_dir / "work" / "feature_store"
    assert route("sample_submit.csv") == Path("sample_submit.csv")
    assert route("artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json") == Path("artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json")


def test_run_paths_are_explicit_and_not_record_names() -> None:
    paths = attempt.run_paths(Path("artifacts") / "isolated")
    assert paths["submission"].name == "candidate_submission.csv"
    assert paths["raw_predictions"].name == "raw_specialist_predictions.parquet"
    assert "submission_specialized_hurdle_joint_rmsle.csv" not in {path.name for path in paths.values()}


def test_immutable_hash_contract_rejects_drift(monkeypatch) -> None:
    monkeypatch.setattr(attempt, "INPUT_PATHS", {"sample_template": Path("sample_submit.csv")})
    monkeypatch.setattr(attempt, "EXPECTED_INPUT_HASHES", {"sample_template": "0" * 64})
    try:
        attempt.immutable_input_hashes()
    except attempt.ContractError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("expected immutable hash contract to reject drift")
