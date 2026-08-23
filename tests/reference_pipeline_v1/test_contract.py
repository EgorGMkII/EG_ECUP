from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from src.reference_pipeline_v1.contract import POST_NY_PUBLIC_PROXY, anchor_manifest, validate_profile, windows
from src.reference_pipeline_v1.meta import fit_meta, load_predict
from src.reference_pipeline_v1.models import EventTimeTransformer, S1MaskedPretrainer, S2MultiHorizonPretrainer
from src.sequential.dataset import build_user_sequence_tensor


def test_anchors_and_cutoffs() -> None:
    validate_profile(); rows = anchor_manifest()
    assert POST_NY_PUBLIC_PROXY.meta_anchor == "2025-12-15"
    assert POST_NY_PUBLIC_PROXY.validation_anchor == "2026-01-14"
    assert len(POST_NY_PUBLIC_PROXY.run_a_anchors) == 17
    assert len(POST_NY_PUBLIC_PROXY.run_b_anchors) == 20
    assert len(rows) == 39
    assert windows("2025-03-31")["state_history_start"] == "2025-01-01"


def test_final_run_b_target_may_end_on_validation_anchor() -> None:
    assert windows(POST_NY_PUBLIC_PROXY.run_b_anchors[-1])["model_target_end"] == POST_NY_PUBLIC_PROXY.validation_anchor
    validate_profile(POST_NY_PUBLIC_PROXY)


def test_s1_s2_are_not_equivalent() -> None:
    assert S1MaskedPretrainer.implementation_id != S2MultiHorizonPretrainer.implementation_id
    assert set(S2MultiHorizonPretrainer()(torch.zeros(2, 180, 15))) == {"buy_7", "buy_14", "buy_30", "gmv_7", "gmv_14", "gmv_30"}


def test_ett_all_empty_is_finite_and_zero() -> None:
    model = EventTimeTransformer().eval(); content = torch.zeros(3, 180, 12); time = torch.zeros(3, 180, 12); ranks = torch.zeros(3, 180, dtype=torch.long); mask = torch.ones(3, 180, dtype=torch.bool); empty = torch.ones(3, dtype=torch.bool)
    embedding, sequence = model.encode(content, time, ranks, mask, empty); output = model(content, time, ranks, mask, empty)
    assert torch.equal(embedding, torch.zeros_like(embedding)); assert torch.isfinite(sequence).all(); assert all(torch.isfinite(value).all() for value in output.values())


def test_meta_constraints_and_no_double_sigmoid() -> None:
    rng = np.random.default_rng(42); bank = {"react": rng.normal(size=(20, 4)), "churn": rng.normal(size=(20, 4)), "amount": rng.normal(size=(20, 4)), "active": rng.integers(0, 2, 20), "target": rng.random(20)}
    package = fit_meta(bank, "a" * 64); params = np.asarray(package["parameters"])
    assert np.isclose(params[:4].sum(), 1) and np.isclose(params[4:8].sum(), 1)
    pred = load_predict(package, bank); assert pred.shape == (20,) and np.isfinite(pred).all()


def test_daily_tensor_cache_short_circuits_raw_scan() -> None:
    with TemporaryDirectory(prefix="reference_cache_", dir="artifacts") as directory:
        cache_dir = Path(directory)
        cached = np.zeros((2, 180, 15), dtype=np.float32)
        np.save(cache_dir / "seq_tensor_2025-03-31_u2_t180.npy", cached)
        # A cache hit must happen before touching the raw schema.  This deliberately
        # lacks event_date/user_id and would fail if Polars filtering still ran.
        malformed_raw = pl.DataFrame({"not_an_event_column": [1]})
        loaded = build_user_sequence_tensor(malformed_raw, [10, 20], date(2025, 3, 31), seq_len=180, cache_dir=cache_dir)
        assert loaded.shape == cached.shape
        assert isinstance(loaded, np.memmap)
        loaded._mmap.close()
        del loaded
