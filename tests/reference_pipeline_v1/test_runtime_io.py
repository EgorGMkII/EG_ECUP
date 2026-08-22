from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

import scripts.build_joint_rmsle_submission as record_builder
from scripts.run_reference_pipeline_v1 import _masked_loader, build_event_store


def test_event_store_is_pooled_fp16_memmap(monkeypatch) -> None:
    calls: list[str] = []

    def fake_extract(_raw, _users, anchor, *, out_c, out_t, out_r, out_m, out_emp, **_kwargs):
        calls.append(anchor)
        out_c[:] = 1.5
        out_t[:] = 2.5
        out_r[:] = 3
        out_m[:] = False
        out_emp[:] = False

    monkeypatch.setattr(record_builder, "extract_event_time_sequences", fake_extract)
    anchors = ("2025-03-31", "2025-04-14")
    with TemporaryDirectory(prefix="reference_event_store_", dir="artifacts") as directory:
        store = build_event_store(pl.DataFrame(), [10, 20], anchors, Path(directory))

        assert calls == list(anchors)
        assert store.content.dtype == np.float16
        assert store.time.dtype == np.float16
        assert store.rank.dtype == np.int16
        assert store.mask.dtype == np.bool_
        assert store.content.shape == (4, 180, 12)
        assert store.get(anchors[1])[0].shape == (2, 180, 12)
        assert np.all(store.get(anchors[1])[0] == np.float16(1.5))
        store.close()


def test_masked_loader_reads_only_selected_rows() -> None:
    values = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32)
    target = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    batches = list(_masked_loader(values, target, mask=np.asarray([True, False, True]), batch=2))
    observed = sorted(zip(batches[0][0].numpy().ravel().tolist(), batches[0][1].numpy().tolist()))
    assert observed == [(10.0, 1.0), (30.0, 3.0)]
