"""Run-scoped causal stores shared by every SSL V1 first-level model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.features import compute_global_platform_table
from src.sequential.dataset import build_user_sequence_tensor
from src.snapshots import build_snapshot

from .contract import EXPERIMENT, load_feature_order
from .runtime import progress


ALLOWED_CATBOOST_NAN_FEATURES = frozenset({
    "ts_gmv_skewness_90d",
    "ts_gmv_kurtosis_90d",
})


def validate_catboost_feature_values(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    *,
    anchor: str,
) -> dict[str, int]:
    matrix = frame.select(feature_names).to_numpy()
    if np.isinf(matrix).any():
        raise RuntimeError(f"Infinite CatBoost feature found for {anchor}")
    nan_counts: dict[str, int] = {}
    for index, name in enumerate(feature_names):
        count = int(np.isnan(matrix[:, index]).sum())
        if count:
            nan_counts[name] = count
    unexpected = set(nan_counts) - ALLOWED_CATBOOST_NAN_FEATURES
    if unexpected:
        raise RuntimeError(f"Unexpected NaN CatBoost features for {anchor}: {sorted(unexpected)}")
    return nan_counts


def required_anchors() -> tuple[str, ...]:
    """Return the stable 15-anchor store union."""

    return tuple(sorted(set((
        *EXPERIMENT.run_a_anchors,
        EXPERIMENT.meta_anchor,
        *EXPERIMENT.run_b_anchors,
        EXPERIMENT.validation_anchor,
    ))))


def build_state_labels(
    raw: pl.DataFrame,
    users: list[int],
    anchor: str,
    horizon_days: int = 30,
) -> pl.DataFrame:
    """Build state and future labels without leaking them into features."""

    anchor_date = date.fromisoformat(anchor)
    index = pl.DataFrame({"user_id": users})
    user_filter = pl.col("user_id").is_in(users)
    past = (
        raw.filter(
            user_filter
            & pl.col("event_date").is_between(anchor_date - timedelta(days=89), anchor_date)
        )
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("past_gmv_90d"))
    )
    future = (
        raw.filter(
            user_filter
            & pl.col("event_date").is_between(
                anchor_date + timedelta(days=1), anchor_date + timedelta(days=horizon_days)
            )
        )
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("future_gmv_30d"))
    )
    return (
        index.join(past, on="user_id", how="left")
        .join(future, on="user_id", how="left")
        .with_columns(
            pl.col("past_gmv_90d").fill_null(0.0),
            pl.col("future_gmv_30d").fill_null(0.0),
        )
        .with_columns(
            (pl.col("past_gmv_90d") > 0).cast(pl.Int8).alias("was_active"),
            (pl.col("future_gmv_30d") > 0).cast(pl.Int8).alias("will_buy"),
            pl.col("future_gmv_30d").log1p().alias("z_target"),
        )
    )


@dataclass
class FeatureFrameStore:
    root: Path
    anchors: tuple[str, ...]
    users: int
    feature_names: tuple[str, ...]

    def path(self, anchor: str) -> Path:
        if anchor not in self.anchors:
            raise KeyError(f"Unknown feature anchor: {anchor}")
        return self.root / f"frame_{anchor}.parquet"

    def get(self, anchor: str) -> pl.DataFrame:
        frame = pl.read_parquet(self.path(anchor))
        expected = (
            "user_id", *self.feature_names, "was_active", "will_buy",
            "future_gmv_30d", "z_target",
        )
        if frame.height != self.users or tuple(frame.columns) != expected:
            raise RuntimeError(f"Invalid cached feature frame for {anchor}")
        return frame


def build_feature_frame_store(
    raw: pl.DataFrame,
    users: list[int],
    anchors: tuple[str, ...],
    root: Path,
    cohort_sha256: str = EXPERIMENT.cohort_sha256,
) -> FeatureFrameStore:
    root.mkdir(parents=True, exist_ok=True)
    canonical = load_feature_order()
    metadata_path = root / "metadata.json"
    metadata = {
        "anchors": list(anchors),
        "users": len(users),
        "cohort_sha256": cohort_sha256,
        "feature_order_sha256": EXPERIMENT.feature_order_sha256,
        "feature_count": len(canonical),
    }
    paths = [root / f"frame_{anchor}.parquet" for anchor in anchors]
    if metadata_path.exists() and all(path.exists() for path in paths):
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed == metadata:
            store = FeatureFrameStore(root, anchors, len(users), canonical)
            for anchor in anchors:
                store.get(anchor)
            progress("FEATURE_STORE_CACHE_HIT", anchors=len(anchors), users=len(users))
            return store

    global_table = compute_global_platform_table(raw)
    excluded = {
        "user_id", "anchor_date", "history_start", "history_end", "target_start",
        "target_end", "target", "will_buy_30d", "user_segment_id", "lifetime_gmv",
    }
    progress("FEATURE_STORE_START", anchors=len(anchors), users=len(users))
    for index, anchor in enumerate(anchors, start=1):
        snapshot = build_snapshot(
            raw,
            users,
            date.fromisoformat(anchor),
            global_table=global_table,
            history_days=90,
            target_days=30,
        )
        observed_features = tuple(name for name in snapshot.columns if name not in excluded)
        if observed_features != canonical:
            missing = [name for name in canonical if name not in observed_features]
            unexpected = [name for name in observed_features if name not in canonical]
            raise RuntimeError(
                f"CatBoost feature contract mismatch for {anchor}: "
                f"observed={len(observed_features)} missing={missing} unexpected={unexpected}"
            )
        labels = build_state_labels(raw, users, anchor)
        frame = (
            snapshot.select(["user_id", *canonical])
            .join(labels.select(["user_id", "was_active", "will_buy", "future_gmv_30d", "z_target"]), on="user_id", how="inner")
            .select(["user_id", *canonical, "was_active", "will_buy", "future_gmv_30d", "z_target"])
        )
        if frame.height != len(users) or frame["user_id"].to_list() != users:
            raise RuntimeError(f"Feature frame user alignment failed for {anchor}")
        nan_counts = validate_catboost_feature_values(frame, canonical, anchor=anchor)
        frame.write_parquet(root / f"frame_{anchor}.parquet")
        progress(
            "FEATURE_STORE_ANCHOR_DONE", anchor=anchor, index=index,
            total=len(anchors), allowed_nan_counts=nan_counts,
        )
        del snapshot, labels, frame
        gc.collect()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    progress("FEATURE_STORE_DONE", anchors=len(anchors), features=len(canonical))
    return FeatureFrameStore(root, anchors, len(users), canonical)


@dataclass
class DailyTensorStore:
    root: Path
    anchors: tuple[str, ...]
    users: int

    def path(self, anchor: str) -> Path:
        if anchor not in self.anchors:
            raise KeyError(f"Unknown daily anchor: {anchor}")
        return self.root / f"seq_tensor_{anchor}_u{self.users}_t180.npy"

    def get(self, anchor: str) -> np.memmap:
        values = np.load(self.path(anchor), mmap_mode="r")
        if values.shape != (self.users, 180, 15) or values.dtype != np.float32:
            raise RuntimeError(f"Invalid dense daily tensor for {anchor}: {values.shape}/{values.dtype}")
        return values


def build_daily_tensor_store(
    raw: pl.DataFrame,
    users: list[int],
    anchors: tuple[str, ...],
    root: Path,
) -> DailyTensorStore:
    root.mkdir(parents=True, exist_ok=True)
    progress("DAILY_STORE_START", anchors=len(anchors), users=len(users))
    for index, anchor in enumerate(anchors, start=1):
        values = build_user_sequence_tensor(
            raw, users, date.fromisoformat(anchor), seq_len=180, cache_dir=root
        )
        if values.shape != (len(users), 180, 15) or values.dtype != np.float32:
            raise RuntimeError(f"Invalid dense daily tensor for {anchor}: {values.shape}/{values.dtype}")
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del values
        progress("DAILY_STORE_ANCHOR_DONE", anchor=anchor, index=index, total=len(anchors))
    progress("DAILY_STORE_DONE", anchors=len(anchors), users=len(users))
    return DailyTensorStore(root, anchors, len(users))


@dataclass
class HorizonLabelStore:
    root: Path
    anchors: tuple[str, ...]
    users: int

    def get(self, anchor: str) -> dict[int, np.ndarray]:
        if anchor not in self.anchors:
            raise KeyError(f"Unknown horizon anchor: {anchor}")
        with np.load(self.root / f"horizons_{anchor}.npz") as values:
            result = {horizon: values[f"h{horizon}"].copy() for horizon in (7, 14, 30)}
        if any(array.shape != (self.users,) or array.dtype != np.float32 for array in result.values()):
            raise RuntimeError(f"Invalid horizon labels for {anchor}")
        return result


def build_horizon_label_store(
    raw: pl.DataFrame,
    users: list[int],
    anchors: tuple[str, ...],
    root: Path,
) -> HorizonLabelStore:
    root.mkdir(parents=True, exist_ok=True)
    index = pl.DataFrame({"user_id": users})
    user_filter = pl.col("user_id").is_in(users)
    progress("HORIZON_STORE_START", anchors=len(anchors), users=len(users))
    for position, anchor in enumerate(anchors, start=1):
        path = root / f"horizons_{anchor}.npz"
        if not path.exists():
            anchor_date = date.fromisoformat(anchor)
            output: dict[str, np.ndarray] = {}
            for horizon in (7, 14, 30):
                sums = (
                    raw.filter(
                        user_filter
                        & pl.col("event_date").is_between(
                            anchor_date + timedelta(days=1), anchor_date + timedelta(days=horizon)
                        )
                    )
                    .group_by("user_id")
                    .agg(pl.col("gmv").sum().alias("gmv"))
                )
                output[f"h{horizon}"] = (
                    index.join(sums, on="user_id", how="left")["gmv"]
                    .fill_null(0.0)
                    .to_numpy()
                    .astype(np.float32, copy=False)
                )
            np.savez(path, **output)
        progress("HORIZON_STORE_ANCHOR_DONE", anchor=anchor, index=position, total=len(anchors))
    store = HorizonLabelStore(root, anchors, len(users))
    for anchor in anchors:
        store.get(anchor)
    progress("HORIZON_STORE_DONE", anchors=len(anchors), users=len(users))
    return store


@dataclass
class EventMemmapStore:
    root: Path
    anchors: tuple[str, ...]
    users: int
    offsets: dict[str, int]
    content: np.memmap
    time_features: np.memmap
    ranks: np.memmap
    padding_mask: np.memmap
    empty: np.memmap

    @classmethod
    def open(cls, root: Path, anchors: tuple[str, ...], users: int) -> "EventMemmapStore":
        offsets = {anchor: index * users for index, anchor in enumerate(anchors)}
        total = len(anchors) * users
        arrays = {
            "content": np.load(root / "content.npy", mmap_mode="r"),
            "time_features": np.load(root / "time.npy", mmap_mode="r"),
            "ranks": np.load(root / "rank.npy", mmap_mode="r"),
            "padding_mask": np.load(root / "mask.npy", mmap_mode="r"),
            "empty": np.load(root / "empty.npy", mmap_mode="r"),
        }
        expected = {
            "content": ((total, 180, 12), np.dtype(np.float16)),
            "time_features": ((total, 180, 12), np.dtype(np.float16)),
            "ranks": ((total, 180), np.dtype(np.int16)),
            "padding_mask": ((total, 180), np.dtype(bool)),
            "empty": ((total,), np.dtype(bool)),
        }
        for name, array in arrays.items():
            shape, dtype = expected[name]
            if array.shape != shape or array.dtype != dtype:
                raise RuntimeError(f"Invalid event store {name}: {array.shape}/{array.dtype}")
        return cls(root, anchors, users, offsets, **arrays)

    def get(self, anchor: str) -> tuple[np.ndarray, ...]:
        if anchor not in self.offsets:
            raise KeyError(f"Unknown event anchor: {anchor}")
        start = self.offsets[anchor]
        stop = start + self.users
        return (
            self.content[start:stop], self.time_features[start:stop], self.ranks[start:stop],
            self.padding_mask[start:stop], self.empty[start:stop],
        )

    def close(self) -> None:
        for array in (self.content, self.time_features, self.ranks, self.padding_mask, self.empty):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def build_event_memmap_store(
    raw: pl.DataFrame,
    users: list[int],
    anchors: tuple[str, ...],
    root: Path,
    cohort_sha256: str = EXPERIMENT.cohort_sha256,
) -> EventMemmapStore:
    from scripts.build_joint_rmsle_submission import extract_event_time_sequences

    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    metadata = {
        "anchors": list(anchors), "users": len(users), "max_events": 180,
        "cohort_sha256": cohort_sha256, "dtype": "float16",
    }
    required = [root / name for name in ("content.npy", "time.npy", "rank.npy", "mask.npy", "empty.npy")]
    if metadata_path.exists() and all(path.exists() for path in required):
        if json.loads(metadata_path.read_text(encoding="utf-8")) == metadata:
            progress("EVENT_STORE_CACHE_HIT", anchors=len(anchors), users=len(users))
            return EventMemmapStore.open(root, anchors, len(users))

    total = len(anchors) * len(users)
    content = np.lib.format.open_memmap(root / "content.npy", mode="w+", dtype=np.float16, shape=(total, 180, 12))
    time_features = np.lib.format.open_memmap(root / "time.npy", mode="w+", dtype=np.float16, shape=(total, 180, 12))
    ranks = np.lib.format.open_memmap(root / "rank.npy", mode="w+", dtype=np.int16, shape=(total, 180))
    padding_mask = np.lib.format.open_memmap(root / "mask.npy", mode="w+", dtype=bool, shape=(total, 180))
    empty = np.lib.format.open_memmap(root / "empty.npy", mode="w+", dtype=bool, shape=(total,))
    user_array = np.asarray(users)
    event_raw = raw.filter(pl.col("user_id").is_in(users))
    progress("EVENT_STORE_START", anchors=len(anchors), users=len(users), dtype="float16")
    for index, anchor in enumerate(anchors):
        start = index * len(users)
        stop = start + len(users)
        content[start:stop] = 0
        time_features[start:stop] = 0
        ranks[start:stop] = 0
        padding_mask[start:stop] = True
        empty[start:stop] = True
        extract_event_time_sequences(
            event_raw, user_array, anchor, out_c=content[start:stop], out_t=time_features[start:stop],
            out_r=ranks[start:stop], out_m=padding_mask[start:stop], out_emp=empty[start:stop],
        )
        progress("EVENT_STORE_ANCHOR_DONE", anchor=anchor, index=index + 1, total=len(anchors))
    for array in (content, time_features, ranks, padding_mask, empty):
        array.flush()
    del content, time_features, ranks, padding_mask, empty, event_raw
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    progress("EVENT_STORE_DONE", anchors=len(anchors), users=len(users))
    return EventMemmapStore.open(root, anchors, len(users))


@dataclass
class StoreRegistry:
    frames: FeatureFrameStore
    daily: DailyTensorStore | None
    horizons: HorizonLabelStore | None
    events: EventMemmapStore | None

    def close(self) -> None:
        if self.events is not None:
            self.events.close()


def build_store_registry(raw: pl.DataFrame, users: list[int], root: Path) -> StoreRegistry:
    return build_store_registry_for_anchors(
        raw,
        users,
        store_anchors=required_anchors(),
        training_anchors=EXPERIMENT.run_b_anchors,
        root=root,
        cohort_sha256=EXPERIMENT.cohort_sha256,
    )


def build_store_registry_for_anchors(
    raw: pl.DataFrame,
    users: list[int],
    *,
    store_anchors: tuple[str, ...],
    training_anchors: tuple[str, ...],
    root: Path,
    cohort_sha256: str,
    required_stores: frozenset[str] | None = None,
) -> StoreRegistry:
    anchors = tuple(sorted(set(store_anchors)))
    if not set(training_anchors).issubset(anchors):
        raise ValueError("Training anchors must be included in the store union")
    root.mkdir(parents=True, exist_ok=True)
    required = required_stores or frozenset({"frames", "daily", "horizons", "events"})
    if "frames" not in required:
        raise ValueError("Feature frames are required by every current adapter")
    frames = build_feature_frame_store(
        raw, users, anchors, root / "frames", cohort_sha256=cohort_sha256
    )
    daily = build_daily_tensor_store(raw, users, anchors, root / "daily") if "daily" in required else None
    horizons = build_horizon_label_store(raw, users, training_anchors, root / "horizons") if "horizons" in required else None
    events = build_event_memmap_store(raw, users, anchors, root / "events", cohort_sha256=cohort_sha256) if "events" in required else None
    return StoreRegistry(frames=frames, daily=daily, horizons=horizons, events=events)
