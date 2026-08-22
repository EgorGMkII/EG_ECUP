"""Run the new REFERENCE_PIPELINE_V1 PRE_NY_PRIMARY baseline in DataSphere.

This is intentionally separate from all forensic/historical submission scripts.
It builds every training frame from raw events inside the job and never writes a
submission.  RUN B receives only the serialized frozen meta package from RUN A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from src.reference_pipeline_v1.contract import CHANNELS, COHORT_SHA256, PRE_NY_PRIMARY, anchor_manifest, cohort_sha256, derive_seed, validate_profile
from src.reference_pipeline_v1.meta import AMOUNT_COLUMNS, CHURN_COLUMNS, REACT_COLUMNS, fit_meta, load_predict
from src.reference_pipeline_v1.models import EventTimeTransformer, S1MaskedPretrainer, S2MultiHorizonPretrainer, Specialist, TransitionBase, transition_loss
from src.sequential.dataset import build_user_sequence_tensor
from src.snapshots import build_snapshot, compute_global_platform_table


ROOT = Path(".")
OUT = Path("artifacts/reference_pipeline_v1/pre_ny_primary")
PREDICTION_COLUMNS = ("user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "transition", *REACT_COLUMNS, *CHURN_COLUMNS, *AMOUNT_COLUMNS)


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


def build_frames(raw: pl.DataFrame, users: list[int], anchors: tuple[str, ...], cache: Path) -> tuple[list[pl.DataFrame], list[str]]:
    global_table = compute_global_platform_table(raw); frames: list[pl.DataFrame] = []; feature_names: list[str] | None = None
    excluded = {"user_id", "anchor_date", "history_start", "history_end", "target_start", "target_end", "target", "will_buy_30d", "user_segment_id", "lifetime_gmv"}
    for anchor in anchors:
        snapshot = build_snapshot(raw, users, date.fromisoformat(anchor), global_table=global_table)
        state = labels(raw, users, anchor)
        frame = snapshot.drop([x for x in ("target", "will_buy_30d") if x in snapshot.columns]).join(state, on="user_id", how="inner")
        if feature_names is None: feature_names = [name for name in snapshot.columns if name not in excluded]
        missing = set(feature_names) - set(frame.columns)
        if missing: raise ValueError(f"Feature mismatch for {anchor}: {sorted(missing)}")
        frames.append(frame); frame.select(["user_id", *feature_names, "was_active", "will_buy", "future_gmv_30d", "z_target"]).write_parquet(cache / f"frame_{anchor}.parquet")
    return frames, feature_names or []


def save_manifest(out: Path, profile: str, feature_names: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"reference_pipeline_version": "REFERENCE_PIPELINE_V1", "baseline_kind": "new reference baseline", "profile": profile, "commit_sha": git_sha(), "cohort_sha256": COHORT_SHA256, "anchors": anchor_manifest(), "channels": list(CHANNELS)}
    if feature_names is not None:
        encoded = json.dumps(feature_names, separators=(",", ":")).encode(); manifest["catboost_features"] = {"ordered": feature_names, "count": len(feature_names), "dtype": "float32", "sha256": hashlib.sha256(encoded).hexdigest()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def catboost_predictions(train: list[pl.DataFrame], holdout: pl.DataFrame, features: list[str], run: str) -> dict[str, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    pooled = pl.concat(train); x = pooled.select(features).to_numpy().astype(np.float32); xv = holdout.select(features).to_numpy().astype(np.float32)
    output: dict[str, np.ndarray] = {}
    for task, subset, target, model in (
        ("react", pooled.filter(pl.col("was_active") == 0), "will_buy", CatBoostClassifier),
        ("churn", pooled.filter(pl.col("was_active") == 1), "will_buy", CatBoostClassifier),
        ("amount", pooled.filter(pl.col("future_gmv_30d") > 0), "z_target", CatBoostRegressor),
    ):
        seed = seed_everything(run, "catboost", task); kwargs = dict(iterations=1500, learning_rate=.04, depth=6, l2_leaf_reg=6., random_seed=seed, task_type="GPU", allow_writing_files=False, verbose=False)
        y = subset[target].to_numpy();
        if task == "churn": y = 1 - y
        fitted = model(**kwargs).fit(subset.select(features).to_numpy().astype(np.float32), y)
        if task == "amount": output["cb_amount_z"] = np.clip(fitted.predict(xv), 0, None)
        else: output[f"cb_{task}_logit"] = fitted.predict(xv, prediction_type="RawFormulaVal")
    return output


def _loader(*arrays: np.ndarray, batch: int, shuffle: bool = True) -> DataLoader:
    tensors = [torch.as_tensor(array) for array in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def _float32(series: pl.Series) -> np.ndarray:
    return series.to_numpy().astype(np.float32, copy=False)


def _daily(raw: pl.DataFrame, users: list[int], anchor: str, cache: Path) -> np.ndarray:
    return np.asarray(build_user_sequence_tensor(raw, users, date.fromisoformat(anchor), seq_len=180, cache_dir=cache), dtype=np.float32)


def _horizon_labels(raw: pl.DataFrame, users: list[int], anchor: str) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    value = date.fromisoformat(anchor); index = pl.DataFrame({"user_id": users})
    for horizon in (7, 14, 30):
        sums = raw.filter(pl.col("user_id").is_in(users) & pl.col("event_date").is_between(value + timedelta(days=1), value + timedelta(days=horizon))).group_by("user_id").agg(pl.col("gmv").sum().alias("gmv"))
        result[horizon] = index.join(sums, on="user_id", how="left")["gmv"].fill_null(0.).to_numpy().astype(np.float32)
    return result


def _pretrain_gru(model: S1MaskedPretrainer | S2MultiHorizonPretrainer, kind: str, raw: pl.DataFrame, users: list[int], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str) -> None:
    seed_everything(run, kind, "ssl"); model.to(device).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _epoch in range(4):
        for anchor in anchors:
            daily = _daily(raw, users, anchor, cache)
            if kind == "s1":
                for (x,) in _loader(daily, batch=2048):
                    x = x.to(device); corrupted, mask = model.corrupt(x); loss = model.loss(model(corrupted), x, mask)
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            else:
                horizon = _horizon_labels(raw, users, anchor)
                for x, y7, y14, y30 in _loader(daily, horizon[7], horizon[14], horizon[30], batch=2048):
                    x = x.to(device); future = {h: y.to(device) for h, y in zip((7, 14, 30), (y7, y14, y30))}; buy = {h: (value > 0).float() for h, value in future.items()}; z = {h: torch.log1p(value) for h, value in future.items()}
                    loss = model.loss(model(x), buy, z); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()


def _base_gru(pretrainer: S1MaskedPretrainer | S2MultiHorizonPretrainer, raw: pl.DataFrame, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str) -> TransitionBase:
    encoder = pretrainer.encoder; base = TransitionBase(encoder, lambda x: encoder(x)[1]).to(device); seed_everything(run, "gru", "base")
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        base.train()
        for anchor, frame in zip(anchors, frames):
            arrays = (_daily(raw, users, anchor, cache), _float32(frame["z_target"]), _float32(frame["was_active"]), _float32(frame["will_buy"]))
            for x, z, active, buy in _loader(*arrays, batch=2048):
                output = base(x.to(device)); loss = transition_loss(output, z.to(device), active.to(device), buy.to(device)); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0); optimizer.step()
        scheduler.step(); torch.save({"epoch": epoch + 1, "state_dict": base.state_dict()}, OUT / f"{run}_gru_base_epoch_{epoch + 1}.pt")
    return base


def _gru_specialists(base: TransitionBase, raw: pl.DataFrame, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str, kind: str) -> dict[str, Specialist]:
    result = {task: Specialist(base.encoder, task, kind).to(device) for task in ("react", "churn", "amount")}
    for task, model in result.items():
        seed_everything(run, kind, task); model.freeze_phase_h()
        for steps, lr, phase in ((1500, 1e-3, "H"), (2500, 1e-4, "F")):
            if phase == "F": model.unfreeze_phase_f()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
            completed = 0
            for anchor, frame in zip(anchors, frames):
                daily = _daily(raw, users, anchor, cache); mask = frame["was_active"].to_numpy() == (0 if task == "react" else 1) if task != "amount" else frame["future_gmv_30d"].to_numpy() > 0
                target = _float32(frame["will_buy"]) if task == "react" else (1 - _float32(frame["will_buy"]) if task == "churn" else _float32(frame["z_target"]))
                for x, y in _loader(daily[mask], target[mask], batch=512):
                    prediction = model(x.to(device)); loss = F.binary_cross_entropy_with_logits(prediction, y.to(device)) if task != "amount" else F.mse_loss(prediction, y.to(device)); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); completed += 1
                    if completed >= steps: break
                if completed >= steps: break
            torch.save({"run": run, "task": task, "phase": phase, "state_dict": model.state_dict()}, OUT / f"{run}_{kind}_{task}_{phase}.pt")
    return result


@torch.no_grad()
def _gru_predict(specialists: dict[str, Specialist], raw: pl.DataFrame, users: list[int], anchor: str, cache: Path, device: torch.device, prefix: str) -> dict[str, np.ndarray]:
    daily = _daily(raw, users, anchor, cache); output = {name: [] for name in specialists}
    for (x,) in _loader(daily, batch=2048, shuffle=False):
        for task, model in specialists.items(): output[task].append(model.eval()(x.to(device)).cpu().numpy())
    return {f"{prefix}_{task}_logit" if task != "amount" else f"{prefix}_amount_z": np.concatenate(values) for task, values in output.items()}


def _event(raw: pl.DataFrame, users: list[int], anchor: str, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"ett_{anchor}.npz"
    if path.exists():
        values = np.load(path); return values["content"], values["time"], values["rank"], values["mask"], values["empty"]
    # The canonical record builder remains the single source of the 12+12 feature schema.
    from scripts.build_joint_rmsle_submission import extract_event_time_sequences
    n = len(users); content = np.zeros((n, 180, 12), dtype=np.float32); time = np.zeros((n, 180, 12), dtype=np.float32); rank = np.zeros((n, 180), dtype=np.int64); mask = np.ones((n, 180), dtype=bool); empty = np.ones(n, dtype=bool)
    extract_event_time_sequences(raw, np.asarray(users), anchor, out_c=content, out_t=time, out_r=rank, out_m=mask, out_emp=empty)
    np.savez_compressed(path, content=content, time=time, rank=rank, mask=mask, empty=empty)
    return content, time, rank, mask, empty


def _base_ett(raw: pl.DataFrame, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str) -> EventTimeTransformer:
    seed_everything(run, "ett", "base"); model = EventTimeTransformer().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # AMP is activated only after an explicit finite forward/backward smoke step.
    warmup, total, micro_step, optimizer_step = 500, 4500, 0, 0; scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total - warmup)
    probe = model(torch.zeros(2, 180, 12, device=device), torch.zeros(2, 180, 12, device=device), torch.zeros(2, 180, dtype=torch.long, device=device), torch.ones(2, 180, dtype=torch.bool, device=device), torch.ones(2, dtype=torch.bool, device=device)); smoke_loss = sum(value.float().square().mean() for value in probe.values()); smoke_loss.backward(); optimizer.zero_grad(); amp = torch.cuda.is_available(); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    while optimizer_step < total:
        for anchor, frame in zip(anchors, frames):
            arrays = (*_event(raw, users, anchor, cache), _float32(frame["z_target"]), _float32(frame["was_active"]), _float32(frame["will_buy"]))
            for c, t, r, m, e, z, active, buy in _loader(*arrays, batch=128):
                with torch.amp.autocast("cuda", enabled=amp): output = model(c.to(device), t.to(device), r.to(device), m.to(device), e.to(device)); loss = transition_loss(output, z.to(device), active.to(device), buy.to(device)) / 4
                scaler.scale(loss).backward()
                micro_step += 1
                if micro_step % 4 == 0:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    optimizer_step += 1
                    if optimizer_step <= warmup:
                        for group in optimizer.param_groups: group["lr"] = 3e-4 * optimizer_step / warmup
                    else: scheduler.step()
                    if optimizer_step in (1500, 3000, 4500): torch.save({"step": optimizer_step, "state_dict": model.state_dict()}, OUT / f"{run}_ett_base_step_{optimizer_step}.pt")
                if optimizer_step >= total: return model
    return model


def _ett_specialists(base: EventTimeTransformer, raw: pl.DataFrame, users: list[int], frames: list[pl.DataFrame], anchors: tuple[str, ...], cache: Path, device: torch.device, run: str) -> dict[str, Specialist]:
    result = {task: Specialist(base, task, "ett").to(device) for task in ("react", "churn", "amount")}
    for task, model in result.items():
        seed_everything(run, "ett", task); model.freeze_phase_h()
        for steps, lr, phase in ((1500, 1e-3, "H"), (2500, 1e-4, "F")):
            if phase == "F": model.unfreeze_phase_f()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps); complete = 0
            for anchor, frame in zip(anchors, frames):
                mask = frame["was_active"].to_numpy() == (0 if task == "react" else 1) if task != "amount" else frame["future_gmv_30d"].to_numpy() > 0
                target = _float32(frame["will_buy"]) if task == "react" else (1 - _float32(frame["will_buy"]) if task == "churn" else _float32(frame["z_target"])); data = _event(raw, users, anchor, cache)
                for batch in _loader(*(array[mask] for array in data), target[mask], batch=512):
                    *inputs, y = batch; prediction = model(*(x.to(device) for x in inputs)); loss = F.binary_cross_entropy_with_logits(prediction, y.to(device)) if task != "amount" else F.mse_loss(prediction, y.to(device)); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); complete += 1
                    if complete >= steps: break
                if complete >= steps: break
            torch.save({"run": run, "task": task, "phase": phase, "state_dict": model.state_dict()}, OUT / f"{run}_ett_{task}_{phase}.pt")
    return result


@torch.no_grad()
def _ett_predict(specialists: dict[str, Specialist], raw: pl.DataFrame, users: list[int], anchor: str, cache: Path, device: torch.device) -> dict[str, np.ndarray]:
    output = {task: [] for task in specialists}
    for batch in _loader(*_event(raw, users, anchor, cache), batch=128, shuffle=False):
        for task, model in specialists.items(): output[task].append(model.eval()(*(x.to(device) for x in batch)).cpu().numpy())
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
    validate_profile(); OUT.mkdir(parents=True, exist_ok=True)
    users_path = Path(args.cohort); actual = cohort_sha256(users_path)
    if actual != COHORT_SHA256: raise RuntimeError(f"Cohort SHA mismatch: {actual}")
    users = pl.read_parquet(users_path)["user_id"].to_list()
    if len(users) != 100_000 or len(set(users)) != 100_000: raise RuntimeError("Cohort must contain 100,000 unique users")
    raw = pl.read_parquet(args.train); cache = OUT / "frames"; cache.mkdir(exist_ok=True)
    if args.smoke:
        probe_users = users[:100]; probe = labels(raw, probe_users, PRE_NY_PRIMARY.run_a_anchors[0])
        if probe.height != 100 or set(("past_gmv_90d", "future_gmv_30d", "was_active", "will_buy", "z_target")) - set(probe.columns):
            raise RuntimeError("V1 smoke label contract failed")
        event_model = EventTimeTransformer().eval(); empty = torch.ones(2, dtype=torch.bool); values = event_model(torch.zeros(2, 180, 12), torch.zeros(2, 180, 12), torch.zeros(2, 180, dtype=torch.long), torch.ones(2, 180, dtype=torch.bool), empty)
        if not all(torch.isfinite(value).all() for value in values.values()): raise RuntimeError("ETT empty-history smoke failed")
        print("REFERENCE_PIPELINE_V1 smoke passed: 100 users, causal state/target labels, finite empty ETT")
        return
    save_manifest(OUT, PRE_NY_PRIMARY.name)
    a_train, features = build_frames(raw, users, PRE_NY_PRIMARY.run_a_anchors, cache); m_frame, _ = build_frames(raw, users, (PRE_NY_PRIMARY.meta_anchor,), cache)
    save_manifest(OUT, PRE_NY_PRIMARY.name, features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); daily_cache, event_cache = OUT / "daily_cache", OUT / "event_cache"; daily_cache.mkdir(exist_ok=True); event_cache.mkdir(exist_ok=True)
    a_predictions = catboost_predictions(a_train, m_frame[0], features, "RUN_A")
    for kind, pretrainer in (("s1", S1MaskedPretrainer()), ("s2", S2MultiHorizonPretrainer())):
        _pretrain_gru(pretrainer, kind, raw, users, PRE_NY_PRIMARY.run_a_anchors, daily_cache, device, "RUN_A"); base = _base_gru(pretrainer, raw, users, a_train, PRE_NY_PRIMARY.run_a_anchors, daily_cache, device, "RUN_A"); a_predictions.update(_gru_predict(_gru_specialists(base, raw, users, a_train, PRE_NY_PRIMARY.run_a_anchors, daily_cache, device, "RUN_A", kind), raw, users, PRE_NY_PRIMARY.meta_anchor, daily_cache, device, kind))
    ett = _base_ett(raw, users, a_train, PRE_NY_PRIMARY.run_a_anchors, event_cache, device, "RUN_A"); a_predictions.update(_ett_predict(_ett_specialists(ett, raw, users, a_train, PRE_NY_PRIMARY.run_a_anchors, event_cache, device, "RUN_A"), raw, users, PRE_NY_PRIMARY.meta_anchor, event_cache, device))
    a_bank = make_bank(m_frame[0], PRE_NY_PRIMARY.meta_anchor, a_predictions); a_path = OUT / "run_a_meta_prediction_bank.parquet"; a_bank.write_parquet(a_path); package = fit_meta(bank_arrays(a_bank), sha(a_path)); (OUT / "frozen_meta_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    b_train, features_b = build_frames(raw, users, PRE_NY_PRIMARY.run_b_anchors, cache); v_frame, _ = build_frames(raw, users, (PRE_NY_PRIMARY.validation_anchor,), cache)
    if features_b != features: raise RuntimeError("RUN B feature manifest differs from RUN A")
    b_predictions = catboost_predictions(b_train, v_frame[0], features, "RUN_B")
    for kind, pretrainer in (("s1", S1MaskedPretrainer()), ("s2", S2MultiHorizonPretrainer())):
        _pretrain_gru(pretrainer, kind, raw, users, PRE_NY_PRIMARY.run_b_anchors, daily_cache, device, "RUN_B"); base = _base_gru(pretrainer, raw, users, b_train, PRE_NY_PRIMARY.run_b_anchors, daily_cache, device, "RUN_B"); b_predictions.update(_gru_predict(_gru_specialists(base, raw, users, b_train, PRE_NY_PRIMARY.run_b_anchors, daily_cache, device, "RUN_B", kind), raw, users, PRE_NY_PRIMARY.validation_anchor, daily_cache, device, kind))
    ett = _base_ett(raw, users, b_train, PRE_NY_PRIMARY.run_b_anchors, event_cache, device, "RUN_B"); b_predictions.update(_ett_predict(_ett_specialists(ett, raw, users, b_train, PRE_NY_PRIMARY.run_b_anchors, event_cache, device, "RUN_B"), raw, users, PRE_NY_PRIMARY.validation_anchor, event_cache, device))
    b_bank = make_bank(v_frame[0], PRE_NY_PRIMARY.validation_anchor, b_predictions); b_path = OUT / "run_b_validation_prediction_bank.parquet"; b_bank.write_parquet(b_path); z = load_predict(package, bank_arrays(b_bank)); report(b_bank, z, OUT / "validation_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--train", default="data/train.parquet"); parser.add_argument("--cohort", default="selected_users_100k.parquet"); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
