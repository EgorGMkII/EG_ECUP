"""Run the new REFERENCE_PIPELINE_V1 POST_NY_PUBLIC_PROXY baseline in DataSphere.

This is intentionally separate from all forensic/historical submission scripts.
It builds every training frame from raw events inside the job and never writes a
submission.  RUN B receives only the serialized frozen meta package from RUN A.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Supports both `python scripts/...` locally and the DataSphere PYTHONPATH entrypoint.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from src.reference_pipeline_v1.contract import CHANNELS, COHORT_SHA256, POST_NY_PUBLIC_PROXY, anchor_manifest, cohort_sha256, derive_seed, validate_profile
from src.reference_pipeline_v1.meta import AMOUNT_COLUMNS, CHURN_COLUMNS, REACT_COLUMNS, fit_meta, load_predict
from src.reference_pipeline_v1.models import EventTimeTransformer, S1MaskedPretrainer, S2MultiHorizonPretrainer, Specialist, TransitionBase, transition_loss
from src.sequential.dataset import build_user_sequence_tensor
from src.snapshots import build_snapshot, compute_global_platform_table


ROOT = Path(".")
ACTIVE_PROFILE = POST_NY_PUBLIC_PROXY
OUT = Path("artifacts/reference_pipeline_v1/post_ny_public_proxy")
PREDICTION_COLUMNS = ("user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "transition", *REACT_COLUMNS, *CHURN_COLUMNS, *AMOUNT_COLUMNS)
PROCESS_STARTED = time.perf_counter()


def progress(stage: str, **details: object) -> None:
    payload = {"stage": stage, "elapsed_seconds": round(time.perf_counter() - PROCESS_STARTED, 1), **details}
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def gpu_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return f"{torch.cuda.get_device_name(0)}, memory_allocated={torch.cuda.memory_allocated(0)}"


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): hasher.update(block)
    return hasher.hexdigest()


def git_sha() -> str:
    declared = os.environ.get("REFERENCE_V1_COMMIT_SHA")
    if declared:
        return declared
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def seed_everything(run: str, architecture: str, task: str) -> int:
    value = derive_seed(run, architecture, task); random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    return value


def labels(raw: pl.DataFrame, users: list[int], anchor: str, horizon: int = 30) -> pl.DataFrame:
    value = date.fromisoformat(anchor); index = pl.DataFrame({"user_id": users})
    past = raw.filter(pl.col("user_id").is_in(users) & pl.col("event_date").is_between(value - timedelta(days=89), value)).group_by("user_id").agg(pl.col("gmv").sum().alias("past_gmv_90d"))
    future = raw.filter(pl.col("user_id").is_in(users) & pl.col("event_date").is_between(value + timedelta(days=1), value + timedelta(days=horizon))).group_by("user_id").agg(pl.col("gmv").sum().alias("future_gmv_30d"))
    return index.join(past, on="user_id", how="left").join(future, on="user_id", how="left").with_columns(pl.col("past_gmv_90d").fill_null(0.), pl.col("future_gmv_30d").fill_null(0.)).with_columns((pl.col("past_gmv_90d") > 0).cast(pl.Int8).alias("was_active"), (pl.col("future_gmv_30d") > 0).cast(pl.Int8).alias("will_buy"), pl.col("future_gmv_30d").log1p().alias("z_target"))


def build_frame_store(raw: pl.DataFrame, users: list[int], anchors: tuple[str, ...], cache: Path) -> tuple[dict[str, Path], list[str]]:
    """Build every unique causal anchor frame once on the VM and persist it."""
    global_table = compute_global_platform_table(raw); paths: dict[str, Path] = {}; feature_names: list[str] | None = None
    excluded = {"user_id", "anchor_date", "history_start", "history_end", "target_start", "target_end", "target", "will_buy_30d", "user_segment_id", "lifetime_gmv"}
    unique_anchors = tuple(dict.fromkeys(anchors))
    progress("FEATURE_STORE_START", anchors=len(unique_anchors), users=len(users))
    for index, anchor in enumerate(unique_anchors, start=1):
        snapshot = build_snapshot(raw, users, date.fromisoformat(anchor), global_table=global_table)
        state = labels(raw, users, anchor)
        frame = snapshot.drop([x for x in ("target", "will_buy_30d") if x in snapshot.columns]).join(state, on="user_id", how="inner")
        if feature_names is None: feature_names = [name for name in snapshot.columns if name not in excluded]
        missing = set(feature_names) - set(frame.columns)
        if missing: raise ValueError(f"Feature mismatch for {anchor}: {sorted(missing)}")
        path = cache / f"frame_{anchor}.parquet"
        frame.select(["user_id", *feature_names, "was_active", "will_buy", "future_gmv_30d", "z_target"]).write_parquet(path)
        paths[anchor] = path
        progress("FEATURE_STORE_ANCHOR_DONE", anchor=anchor, index=index, total=len(unique_anchors), rows=frame.height)
        del snapshot, state, frame
        gc.collect()
    progress("FEATURE_STORE_DONE", anchors=len(unique_anchors), features=len(feature_names or []))
    return paths, feature_names or []


def load_frames(paths: dict[str, Path], anchors: tuple[str, ...]) -> list[pl.DataFrame]:
    return [pl.read_parquet(paths[anchor]) for anchor in anchors]


def save_manifest(out: Path, profile: str, feature_names: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"reference_pipeline_version": "REFERENCE_PIPELINE_V1", "baseline_kind": "new reference baseline", "profile": profile, "commit_sha": git_sha(), "cohort_sha256": COHORT_SHA256, "anchors": anchor_manifest(), "channels": list(CHANNELS)}
    if feature_names is not None:
        encoded = json.dumps(feature_names, separators=(",", ":")).encode(); manifest["catboost_features"] = {"ordered": feature_names, "count": len(feature_names), "dtype": "float32", "sha256": hashlib.sha256(encoded).hexdigest()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def catboost_predictions(train: list[pl.DataFrame], holdout: pl.DataFrame, features: list[str], run: str) -> dict[str, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    pooled = pl.concat(train); holdout_pool = Pool(holdout.select(features).to_pandas())
    output: dict[str, np.ndarray] = {}
    for task, predicate, target, model in (
        ("react", pl.col("was_active") == 0, "will_buy", CatBoostClassifier),
        ("churn", pl.col("was_active") == 1, "will_buy", CatBoostClassifier),
        ("amount", pl.col("future_gmv_30d") > 0, "z_target", CatBoostRegressor),
    ):
        subset = pooled.filter(predicate)
        progress("CATBOOST_TASK_START", run=run, task=task, rows=subset.height, gpu=gpu_snapshot())
        seed = seed_everything(run, "catboost", task); kwargs = dict(iterations=1500, learning_rate=.04, depth=6, l2_leaf_reg=6., random_seed=seed, task_type="GPU", allow_writing_files=False, verbose=250)
        y = subset[target].to_numpy();
        if task == "churn": y = 1 - y
        train_pool = Pool(subset.select(features).to_pandas(), label=y)
        fitted = model(**kwargs).fit(train_pool)
        if task == "amount": output["cb_amount_z"] = np.clip(fitted.predict(holdout_pool), 0, None)
        else: output[f"cb_{task}_logit"] = fitted.predict(holdout_pool, prediction_type="RawFormulaVal")
        progress("CATBOOST_TASK_DONE", run=run, task=task, gpu=gpu_snapshot())
        del train_pool, fitted, subset, y
        gc.collect()
    del pooled, holdout_pool
    gc.collect()
    return output


def _loader(*arrays: np.ndarray, batch: int, shuffle: bool = True) -> DataLoader:
    tensors = [torch.as_tensor(array) for array in arrays]
    workers = 2 if torch.cuda.is_available() and os.name != "nt" else 0
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _masked_loader(*arrays: np.ndarray, mask: np.ndarray, batch: int) -> DataLoader:
    """Read selected memmap rows without materializing boolean-index copies."""
    tensors = [torch.as_tensor(array) for array in arrays]
    dataset = Subset(TensorDataset(*tensors), np.flatnonzero(mask))
    workers = 2 if torch.cuda.is_available() and os.name != "nt" else 0
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _float32(series: pl.Series) -> np.ndarray:
    return series.to_numpy().astype(np.float32, copy=False)


def _daily(raw: pl.DataFrame | None, users: list[int], anchor: str, cache: Path) -> np.ndarray:
    path = cache / f"seq_tensor_{anchor}_u{len(users)}_t180.npy"
    was_cached = path.exists()
    if not was_cached: progress("DAILY_TENSOR_BUILD_START", anchor=anchor, users=len(users))
    if raw is None and not was_cached:
        raise RuntimeError(f"Missing prebuilt daily tensor after raw log release: {path}")
    assert raw is not None or was_cached
    values = build_user_sequence_tensor(raw, users, date.fromisoformat(anchor), seq_len=180, cache_dir=cache)
    if not path.exists():
        raise RuntimeError(f"Daily tensor builder did not create expected cache: {path}")
    if isinstance(values, np.memmap):
        result = values
    else:
        result = np.asarray(values, dtype=np.float32)
    if not was_cached:
        progress("DAILY_TENSOR_READY", anchor=anchor, shape=result.shape, dtype=str(result.dtype))
    return result


def _horizon_labels(raw: pl.DataFrame | None, users: list[int], anchor: str, cache: Path) -> dict[int, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"horizons_{anchor}.npz"
    if path.exists():
        with np.load(path) as values:
            return {horizon: values[f"h{horizon}"].copy() for horizon in (7, 14, 30)}
    if raw is None:
        raise RuntimeError(f"Missing prebuilt horizon labels after raw log release: {path}")
    progress("HORIZON_LABELS_BUILD_START", anchor=anchor)
    result: dict[int, np.ndarray] = {}
    value = date.fromisoformat(anchor); index = pl.DataFrame({"user_id": users})
    for horizon in (7, 14, 30):
        sums = raw.filter(pl.col("user_id").is_in(users) & pl.col("event_date").is_between(value + timedelta(days=1), value + timedelta(days=horizon))).group_by("user_id").agg(pl.col("gmv").sum().alias("gmv"))
        result[horizon] = index.join(sums, on="user_id", how="left")["gmv"].fill_null(0.).to_numpy().astype(np.float32)
    np.savez(path, **{f"h{horizon}": values for horizon, values in result.items()})
    progress("HORIZON_LABELS_READY", anchor=anchor)
    return result


def _pretrain_gru(model: S1MaskedPretrainer | S2MultiHorizonPretrainer, kind: str, raw: pl.DataFrame | None, users: list[int], anchors: tuple[str, ...], cache: Path, label_cache: Path, device: torch.device, run: str) -> None:
    seed_everything(run, kind, "ssl"); model.to(device).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    progress("GRU_PRETRAIN_START", run=run, model=kind, epochs=4, gpu=gpu_snapshot())
    for epoch in range(4):
        for anchor in anchors:
            daily = _daily(raw, users, anchor, cache)
            if kind == "s1":
                for (x,) in _loader(daily, batch=2048):
                    x = x.to(device, non_blocking=True); corrupted, mask = model.corrupt(x); loss = model.loss(model(corrupted), x, mask)
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            else:
                horizon = _horizon_labels(raw, users, anchor, label_cache)
                for x, y7, y14, y30 in _loader(daily, horizon[7], horizon[14], horizon[30], batch=2048):
                    x = x.to(device, non_blocking=True); future = {h: y.to(device, non_blocking=True) for h, y in zip((7, 14, 30), (y7, y14, y30))}; buy = {h: (value > 0).float() for h, value in future.items()}; z = {h: torch.log1p(value) for h, value in future.items()}
                    loss = model.loss(model(x), buy, z); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        progress("GRU_PRETRAIN_EPOCH_DONE", run=run, model=kind, epoch=epoch + 1, epochs=4, loss=float(loss.detach().cpu()))
    progress("GRU_PRETRAIN_DONE", run=run, model=kind, gpu=gpu_snapshot())


def _base_gru(pretrainer: S1MaskedPretrainer | S2MultiHorizonPretrainer, raw: pl.DataFrame | None, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str) -> TransitionBase:
    encoder = pretrainer.encoder; base = TransitionBase(encoder, lambda x: encoder(x)[1]).to(device); seed_everything(run, "gru", "base")
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    progress("GRU_BASE_START", run=run, epochs=10, gpu=gpu_snapshot())
    for epoch in range(10):
        base.train()
        for anchor, frame in zip(anchors, frames):
            arrays = (_daily(raw, users, anchor, cache), _float32(frame["z_target"]), _float32(frame["was_active"]), _float32(frame["will_buy"]))
            for x, z, active, buy in _loader(*arrays, batch=2048):
                output = base(x.to(device, non_blocking=True)); loss = transition_loss(output, z.to(device, non_blocking=True), active.to(device, non_blocking=True), buy.to(device, non_blocking=True)); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); optimizer.step()
        scheduler.step(); torch.save({"epoch": epoch + 1, "state_dict": base.state_dict()}, OUT / f"{run}_gru_base_epoch_{epoch + 1}.pt")
        progress("GRU_BASE_EPOCH_DONE", run=run, epoch=epoch + 1, epochs=10, loss=float(loss.detach().cpu()))
    progress("GRU_BASE_DONE", run=run, gpu=gpu_snapshot())
    return base


def _gru_specialists(base: TransitionBase, raw: pl.DataFrame | None, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str, kind: str) -> dict[str, Specialist]:
    result = {task: Specialist(base.encoder, task, kind).to(device) for task in ("react", "churn", "amount")}
    for task, model in result.items():
        seed_everything(run, kind, task); model.freeze_phase_h()
        for steps, lr, phase in ((1500, 1e-3, "H"), (2500, 1e-4, "F")):
            progress("SPECIALIST_PHASE_START", run=run, model=kind, task=task, phase=phase, max_steps=steps, gpu=gpu_snapshot())
            if phase == "F": model.unfreeze_phase_f()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
            completed = 0
            for anchor, frame in zip(anchors, frames):
                daily = _daily(raw, users, anchor, cache); mask = frame["was_active"].to_numpy() == (0 if task == "react" else 1) if task != "amount" else frame["future_gmv_30d"].to_numpy() > 0
                target = _float32(frame["will_buy"]) if task == "react" else (1 - _float32(frame["will_buy"]) if task == "churn" else _float32(frame["z_target"]))
                for x, y in _masked_loader(daily, target, mask=mask, batch=512):
                    prediction = model(x.to(device, non_blocking=True)); target_batch = y.to(device, non_blocking=True); loss = F.binary_cross_entropy_with_logits(prediction, target_batch) if task != "amount" else F.mse_loss(prediction, target_batch); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); completed += 1
                    if completed % 250 == 0: progress("SPECIALIST_PROGRESS", run=run, model=kind, task=task, phase=phase, step=completed, max_steps=steps, loss=float(loss.detach().cpu()))
                    if completed >= steps: break
                if completed >= steps: break
            torch.save({"run": run, "task": task, "phase": phase, "state_dict": model.state_dict()}, OUT / f"{run}_{kind}_{task}_{phase}.pt")
            progress("SPECIALIST_PHASE_DONE", run=run, model=kind, task=task, phase=phase, completed_steps=completed, max_steps=steps, loss=float(loss.detach().cpu()), gpu=gpu_snapshot())
    return result


@torch.no_grad()
def _gru_predict(specialists: dict[str, Specialist], raw: pl.DataFrame | None, users: list[int], anchor: str, cache: Path, device: torch.device, prefix: str) -> dict[str, np.ndarray]:
    daily = _daily(raw, users, anchor, cache); output = {name: [] for name in specialists}
    for (x,) in _loader(daily, batch=2048, shuffle=False):
        device_x = x.to(device, non_blocking=True)
        for task, model in specialists.items(): output[task].append(model.eval()(device_x).cpu().numpy())
    return {f"{prefix}_{task}_logit" if task != "amount" else f"{prefix}_amount_z": np.concatenate(values) for task, values in output.items()}


class EventMemmapStore:
    def __init__(self, cache: Path, anchors: tuple[str, ...], users: int) -> None:
        self.anchors = anchors; self.users = users; self.offsets = {anchor: index * users for index, anchor in enumerate(anchors)}
        self.content = np.load(cache / "content.npy", mmap_mode="r")
        self.time = np.load(cache / "time.npy", mmap_mode="r")
        self.rank = np.load(cache / "rank.npy", mmap_mode="r")
        self.mask = np.load(cache / "mask.npy", mmap_mode="r")
        self.empty = np.load(cache / "empty.npy", mmap_mode="r")

    def get(self, anchor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = self.offsets[anchor]; stop = start + self.users
        return self.content[start:stop], self.time[start:stop], self.rank[start:stop], self.mask[start:stop], self.empty[start:stop]

    def close(self) -> None:
        for array in (self.content, self.time, self.rank, self.mask, self.empty):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def build_event_store(raw: pl.DataFrame, users: list[int], anchors: tuple[str, ...], cache: Path) -> EventMemmapStore:
    """Build one pooled FP16 event-time memmap for every required anchor."""
    from scripts.build_joint_rmsle_submission import extract_event_time_sequences

    cache.mkdir(parents=True, exist_ok=True)
    unique_anchors = tuple(dict.fromkeys(anchors)); metadata_path = cache / "metadata.json"
    expected = {"anchors": list(unique_anchors), "users": len(users), "max_events": 180}
    required = [cache / name for name in ("content.npy", "time.npy", "rank.npy", "mask.npy", "empty.npy")]
    if metadata_path.exists() and all(path.exists() for path in required) and json.loads(metadata_path.read_text(encoding="utf-8")) == expected:
        progress("EVENT_STORE_CACHE_HIT", anchors=len(unique_anchors), users=len(users))
        return EventMemmapStore(cache, unique_anchors, len(users))

    total = len(unique_anchors) * len(users)
    content = np.lib.format.open_memmap(cache / "content.npy", mode="w+", dtype=np.float16, shape=(total, 180, 12))
    time_values = np.lib.format.open_memmap(cache / "time.npy", mode="w+", dtype=np.float16, shape=(total, 180, 12))
    ranks = np.lib.format.open_memmap(cache / "rank.npy", mode="w+", dtype=np.int16, shape=(total, 180))
    masks = np.lib.format.open_memmap(cache / "mask.npy", mode="w+", dtype=bool, shape=(total, 180))
    empty = np.lib.format.open_memmap(cache / "empty.npy", mode="w+", dtype=bool, shape=(total,))
    user_array = np.asarray(users)
    progress("EVENT_STORE_START", anchors=len(unique_anchors), users=len(users), dtype="float16")
    for index, anchor in enumerate(unique_anchors):
        start = index * len(users); stop = start + len(users)
        content[start:stop] = 0; time_values[start:stop] = 0; ranks[start:stop] = 0; masks[start:stop] = True; empty[start:stop] = True
        extract_event_time_sequences(raw, user_array, anchor, out_c=content[start:stop], out_t=time_values[start:stop], out_r=ranks[start:stop], out_m=masks[start:stop], out_emp=empty[start:stop])
        progress("EVENT_STORE_ANCHOR_DONE", anchor=anchor, index=index + 1, total=len(unique_anchors))
    content.flush(); time_values.flush(); ranks.flush(); masks.flush(); empty.flush()
    del content, time_values, ranks, masks, empty
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    progress("EVENT_STORE_DONE", anchors=len(unique_anchors), users=len(users))
    return EventMemmapStore(cache, unique_anchors, len(users))


def _base_ett(store: EventMemmapStore, frames: list[pl.DataFrame], anchors: tuple[str, ...], device: torch.device, run: str) -> EventTimeTransformer:
    seed_everything(run, "ett", "base"); model = EventTimeTransformer().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # AMP is activated only after an explicit finite forward/backward smoke step.
    warmup, total, micro_step, optimizer_step = 500, 4500, 0, 0; scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total - warmup)
    progress("ETT_BASE_START", run=run, optimizer_steps=total, accumulation=4, gpu=gpu_snapshot())
    probe = model(torch.zeros(2, 180, 12, device=device), torch.zeros(2, 180, 12, device=device), torch.zeros(2, 180, dtype=torch.long, device=device), torch.ones(2, 180, dtype=torch.bool, device=device), torch.ones(2, dtype=torch.bool, device=device)); smoke_loss = sum(value.float().square().mean() for value in probe.values()); smoke_loss.backward(); optimizer.zero_grad(); amp = torch.cuda.is_available(); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    while optimizer_step < total:
        for anchor, frame in zip(anchors, frames):
            arrays = (*store.get(anchor), _float32(frame["z_target"]), _float32(frame["was_active"]), _float32(frame["will_buy"]))
            for c, t, r, m, e, z, active, buy in _loader(*arrays, batch=128):
                with torch.amp.autocast("cuda", enabled=amp): output = model(c.to(device, non_blocking=True), t.to(device, non_blocking=True), r.to(device, non_blocking=True).long(), m.to(device, non_blocking=True), e.to(device, non_blocking=True)); loss = transition_loss(output, z.to(device, non_blocking=True), active.to(device, non_blocking=True), buy.to(device, non_blocking=True)) / 4
                scaler.scale(loss).backward()
                micro_step += 1
                if micro_step % 4 == 0:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    optimizer_step += 1
                    if optimizer_step <= warmup:
                        for group in optimizer.param_groups: group["lr"] = 3e-4 * optimizer_step / warmup
                    else: scheduler.step()
                    if optimizer_step in (1500, 3000, 4500): torch.save({"step": optimizer_step, "state_dict": model.state_dict()}, OUT / f"{run}_ett_base_step_{optimizer_step}.pt")
                    if optimizer_step % 250 == 0: progress("ETT_BASE_PROGRESS", run=run, step=optimizer_step, total=total, loss=float(loss.detach().cpu() * 4))
                if optimizer_step >= total:
                    progress("ETT_BASE_DONE", run=run, optimizer_steps=optimizer_step, gpu=gpu_snapshot())
                    return model
    return model


def _ett_specialists(base: EventTimeTransformer, store: EventMemmapStore, frames: list[pl.DataFrame], anchors: tuple[str, ...], device: torch.device, run: str) -> dict[str, Specialist]:
    result = {task: Specialist(base, task, "ett").to(device) for task in ("react", "churn", "amount")}
    for task, model in result.items():
        seed_everything(run, "ett", task); model.freeze_phase_h()
        for steps, lr, phase in ((1500, 1e-3, "H"), (2500, 1e-4, "F")):
            progress("SPECIALIST_PHASE_START", run=run, model="ett", task=task, phase=phase, max_steps=steps, gpu=gpu_snapshot())
            if phase == "F": model.unfreeze_phase_f()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps); complete = 0
            for anchor, frame in zip(anchors, frames):
                mask = frame["was_active"].to_numpy() == (0 if task == "react" else 1) if task != "amount" else frame["future_gmv_30d"].to_numpy() > 0
                target = _float32(frame["will_buy"]) if task == "react" else (1 - _float32(frame["will_buy"]) if task == "churn" else _float32(frame["z_target"])); data = store.get(anchor)
                for batch in _masked_loader(*data, target, mask=mask, batch=512):
                    *inputs, y = batch
                    device_inputs = [x.to(device, non_blocking=True) for x in inputs]
                    device_inputs[0] = device_inputs[0].float(); device_inputs[1] = device_inputs[1].float()
                    device_inputs[2] = device_inputs[2].long()
                    prediction = model(*device_inputs); target_batch = y.to(device, non_blocking=True); loss = F.binary_cross_entropy_with_logits(prediction, target_batch) if task != "amount" else F.mse_loss(prediction, target_batch); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); complete += 1
                    if complete % 250 == 0: progress("SPECIALIST_PROGRESS", run=run, model="ett", task=task, phase=phase, step=complete, max_steps=steps, loss=float(loss.detach().cpu()))
                    if complete >= steps: break
                if complete >= steps: break
            torch.save({"run": run, "task": task, "phase": phase, "state_dict": model.state_dict()}, OUT / f"{run}_ett_{task}_{phase}.pt")
            progress("SPECIALIST_PHASE_DONE", run=run, model="ett", task=task, phase=phase, completed_steps=complete, max_steps=steps, loss=float(loss.detach().cpu()), gpu=gpu_snapshot())
    return result


@torch.no_grad()
def _ett_predict(specialists: dict[str, Specialist], store: EventMemmapStore, anchor: str, device: torch.device) -> dict[str, np.ndarray]:
    output = {task: [] for task in specialists}
    for batch in _loader(*store.get(anchor), batch=128, shuffle=False):
        device_batch_values = [x.to(device, non_blocking=True) for x in batch]
        device_batch_values[0] = device_batch_values[0].float(); device_batch_values[1] = device_batch_values[1].float()
        device_batch_values[2] = device_batch_values[2].long()
        device_batch = tuple(device_batch_values)
        for task, model in specialists.items(): output[task].append(model.eval()(*device_batch).cpu().numpy())
    return {f"ett_{task}_logit" if task != "amount" else "ett_amount_z": np.concatenate(values) for task, values in output.items()}


def make_bank(frame: pl.DataFrame, anchor: str, predictions: dict[str, np.ndarray]) -> pl.DataFrame:
    base = frame.select("user_id", "was_active", "will_buy", "future_gmv_30d").with_columns(pl.lit(anchor).alias("anchor"), (pl.col("was_active").cast(pl.Utf8) + pl.col("will_buy").cast(pl.Utf8)).alias("transition"))
    for name in (*REACT_COLUMNS, *CHURN_COLUMNS, *AMOUNT_COLUMNS):
        if name not in predictions: raise ValueError(f"Missing model prediction {name}")
        base = base.with_columns(pl.Series(name, predictions[name]))
    return base.select(PREDICTION_COLUMNS)


def bank_arrays(bank: pl.DataFrame) -> dict[str, np.ndarray]:
    return {"react": bank.select(REACT_COLUMNS).to_numpy().astype(np.float64), "churn": bank.select(CHURN_COLUMNS).to_numpy().astype(np.float64), "amount": bank.select(AMOUNT_COLUMNS).to_numpy().astype(np.float64), "active": bank["was_active"].to_numpy(), "target": np.log1p(bank["future_gmv_30d"].to_numpy())}


def report(bank: pl.DataFrame, z_prediction: np.ndarray, path: Path) -> dict:
    target = np.log1p(bank["future_gmv_30d"].to_numpy()); squared = (z_prediction - target) ** 2; result = {"mse": float(squared.mean()), "rmsle": float(np.sqrt(squared.mean())), "transitions": {}}
    for transition in ("00", "01", "10", "11"):
        mask = bank["transition"].to_numpy() == transition; values = squared[mask]
        result["transitions"][transition] = {"rows": int(mask.sum()), "mse": float(values.mean()) if len(values) else None, "rmsle": float(np.sqrt(values.mean())) if len(values) else None, "log_bias": float((z_prediction[mask] - target[mask]).mean()) if len(values) else None, "loss_share": float(values.sum() / squared.sum()) if squared.sum() else 0.}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8"); return result


def run(args: argparse.Namespace) -> None:
    validate_profile(ACTIVE_PROFILE); OUT.mkdir(parents=True, exist_ok=True)
    users_path = Path(args.cohort); actual = cohort_sha256(users_path)
    if actual != COHORT_SHA256: raise RuntimeError(f"Cohort SHA mismatch: {actual}")
    users = pl.read_parquet(users_path)["user_id"].to_list()
    if len(users) != 100_000 or len(set(users)) != 100_000: raise RuntimeError("Cohort must contain 100,000 unique users")
    raw = pl.read_parquet(args.train); cache = OUT / "frames"; cache.mkdir(exist_ok=True)
    if args.smoke:
        probe_users = users[:100]; probe = labels(raw, probe_users, ACTIVE_PROFILE.run_a_anchors[0])
        if probe.height != 100 or set(("past_gmv_90d", "future_gmv_30d", "was_active", "will_buy", "z_target")) - set(probe.columns):
            raise RuntimeError("V1 smoke label contract failed")
        event_model = EventTimeTransformer().eval(); empty = torch.ones(2, dtype=torch.bool); values = event_model(torch.zeros(2, 180, 12), torch.zeros(2, 180, 12), torch.zeros(2, 180, dtype=torch.long), torch.ones(2, 180, dtype=torch.bool), empty)
        if not all(torch.isfinite(value).all() for value in values.values()): raise RuntimeError("ETT empty-history smoke failed")
        print("REFERENCE_PIPELINE_V1 smoke passed: 100 users, causal state/target labels, finite empty ETT")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("REFERENCE_PIPELINE_V1 requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda")
    progress(
        "RUNTIME_READY",
        device=str(device),
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_snapshot=gpu_snapshot(),
    )
    save_manifest(OUT, ACTIVE_PROFILE.name)
    all_anchors = tuple(dict.fromkeys((
        *ACTIVE_PROFILE.run_a_anchors,
        ACTIVE_PROFILE.meta_anchor,
        *ACTIVE_PROFILE.run_b_anchors,
        ACTIVE_PROFILE.validation_anchor,
    )))
    progress(
        "ANCHOR_PLAN",
        run_a_train=list(ACTIVE_PROFILE.run_a_anchors),
        meta_anchor=ACTIVE_PROFILE.meta_anchor,
        run_b_train=list(ACTIVE_PROFILE.run_b_anchors),
        validation_anchor=ACTIVE_PROFILE.validation_anchor,
        unique_feature_store_anchors=list(all_anchors),
    )
    frame_paths, features = build_frame_store(raw, users, all_anchors, cache)
    save_manifest(OUT, ACTIVE_PROFILE.name, features)
    daily_cache, horizon_cache, event_cache = OUT / "daily_cache", OUT / "horizon_cache", OUT / "event_memmap"
    daily_cache.mkdir(exist_ok=True); horizon_cache.mkdir(exist_ok=True); event_cache.mkdir(exist_ok=True)
    progress("SHARED_SEQUENCE_STORE_START", anchors=len(all_anchors))
    event_store = build_event_store(raw, users, all_anchors, event_cache)
    for anchor in all_anchors:
        daily_values = _daily(raw, users, anchor, daily_cache)
        if isinstance(daily_values, np.memmap): daily_values._mmap.close()
        del daily_values
    for anchor in ACTIVE_PROFILE.run_b_anchors:
        _horizon_labels(raw, users, anchor, horizon_cache)
    progress("SHARED_SEQUENCE_STORE_DONE", anchors=len(all_anchors))
    del raw
    raw = None
    gc.collect()
    progress("RAW_LOG_RELEASED")

    progress("RUN_A_START", train_anchors=len(ACTIVE_PROFILE.run_a_anchors), meta_anchor=ACTIVE_PROFILE.meta_anchor)
    a_train = load_frames(frame_paths, ACTIVE_PROFILE.run_a_anchors)
    m_frame = load_frames(frame_paths, (ACTIVE_PROFILE.meta_anchor,))
    a_predictions = catboost_predictions(a_train, m_frame[0], features, "RUN_A")
    for kind, pretrainer in (("s1", S1MaskedPretrainer()), ("s2", S2MultiHorizonPretrainer())):
        _pretrain_gru(pretrainer, kind, raw, users, ACTIVE_PROFILE.run_a_anchors, daily_cache, horizon_cache, device, "RUN_A")
        base = _base_gru(pretrainer, raw, users, a_train, ACTIVE_PROFILE.run_a_anchors, daily_cache, device, "RUN_A")
        gru_specialists = _gru_specialists(base, raw, users, a_train, ACTIVE_PROFILE.run_a_anchors, daily_cache, device, "RUN_A", kind)
        a_predictions.update(_gru_predict(gru_specialists, raw, users, ACTIVE_PROFILE.meta_anchor, daily_cache, device, kind))
        del pretrainer, base, gru_specialists
        gc.collect(); torch.cuda.empty_cache()
    ett = _base_ett(event_store, a_train, ACTIVE_PROFILE.run_a_anchors, device, "RUN_A")
    ett_specialists = _ett_specialists(ett, event_store, a_train, ACTIVE_PROFILE.run_a_anchors, device, "RUN_A")
    a_predictions.update(_ett_predict(ett_specialists, event_store, ACTIVE_PROFILE.meta_anchor, device))
    a_bank = make_bank(m_frame[0], ACTIVE_PROFILE.meta_anchor, a_predictions); a_path = OUT / "run_a_meta_prediction_bank.parquet"; a_bank.write_parquet(a_path); package = fit_meta(bank_arrays(a_bank), sha(a_path)); (OUT / "frozen_meta_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    progress("RUN_A_DONE", meta_anchor=ACTIVE_PROFILE.meta_anchor, meta_rmsle=float(np.sqrt(package["objective"])), prediction_bank=str(a_path), gpu=gpu_snapshot())
    del a_train, m_frame, a_predictions, a_bank, ett, ett_specialists
    gc.collect(); torch.cuda.empty_cache()

    progress("RUN_B_START", train_anchors=len(ACTIVE_PROFILE.run_b_anchors), validation_anchor=ACTIVE_PROFILE.validation_anchor)
    b_train = load_frames(frame_paths, ACTIVE_PROFILE.run_b_anchors)
    v_frame = load_frames(frame_paths, (ACTIVE_PROFILE.validation_anchor,))
    b_predictions = catboost_predictions(b_train, v_frame[0], features, "RUN_B")
    for kind, pretrainer in (("s1", S1MaskedPretrainer()), ("s2", S2MultiHorizonPretrainer())):
        _pretrain_gru(pretrainer, kind, raw, users, ACTIVE_PROFILE.run_b_anchors, daily_cache, horizon_cache, device, "RUN_B")
        base = _base_gru(pretrainer, raw, users, b_train, ACTIVE_PROFILE.run_b_anchors, daily_cache, device, "RUN_B")
        gru_specialists = _gru_specialists(base, raw, users, b_train, ACTIVE_PROFILE.run_b_anchors, daily_cache, device, "RUN_B", kind)
        b_predictions.update(_gru_predict(gru_specialists, raw, users, ACTIVE_PROFILE.validation_anchor, daily_cache, device, kind))
        del pretrainer, base, gru_specialists
        gc.collect(); torch.cuda.empty_cache()
    ett = _base_ett(event_store, b_train, ACTIVE_PROFILE.run_b_anchors, device, "RUN_B")
    ett_specialists = _ett_specialists(ett, event_store, b_train, ACTIVE_PROFILE.run_b_anchors, device, "RUN_B")
    b_predictions.update(_ett_predict(ett_specialists, event_store, ACTIVE_PROFILE.validation_anchor, device))
    b_bank = make_bank(v_frame[0], ACTIVE_PROFILE.validation_anchor, b_predictions); b_path = OUT / "run_b_validation_prediction_bank.parquet"; b_bank.write_parquet(b_path); z = load_predict(package, bank_arrays(b_bank)); report(b_bank, z, OUT / "validation_report.json")
    final_rmsle = float(np.sqrt(np.mean((z - np.log1p(b_bank["future_gmv_30d"].to_numpy())) ** 2)))
    progress("RUN_B_DONE", validation_anchor=ACTIVE_PROFILE.validation_anchor, validation_rmsle=final_rmsle, prediction_bank=str(b_path), gpu=gpu_snapshot())
    event_store.close()
    progress("PIPELINE_DONE", validation_rmsle=final_rmsle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--train", default="data/train.parquet"); parser.add_argument("--cohort", default="selected_users_100k.parquet"); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
