"""Deterministic nested 250k/100k/25k cohort construction."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl

from src.ssl_temporal_stack_v1.runtime import git_sha, sha256_file, write_json
from src.ssl_temporal_stack_v1.stores import build_state_labels

from .profiles import TemporalProfile


TRANSITIONS = ("00", "01", "10", "11")


def _ordered_user_hash(users: list[int]) -> str:
    return sha256("\n".join(map(str, users)).encode()).hexdigest()


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    source = sum(counts.values())
    if source < total:
        raise ValueError("Requested cohort exceeds source universe")
    quotas = {key: counts[key] * total / source for key in TRANSITIONS}
    result = {key: int(quotas[key]) for key in TRANSITIONS}
    for key in sorted(TRANSITIONS, key=lambda item: (-(quotas[item] - result[item]), TRANSITIONS.index(item)))[: total - sum(result.values())]:
        result[key] += 1
    return result


def _rank(profile: str, cohort: str, transition: str, user_id: int, seed: int) -> tuple[str, int]:
    value = f"{seed}|{profile}|{cohort}|{transition}|{user_id}".encode()
    return sha256(value).hexdigest(), user_id


def _manifest(*, profile: TemporalProfile, cohort_name: str, users: list[int], transitions: pl.DataFrame, source_users: list[int], source_sha: str, seed: int) -> dict[str, Any]:
    counts = {key: int(transitions.filter(pl.col("transition") == key).height) for key in TRANSITIONS}
    return {
        "profile": profile.name, "stratification_anchor": profile.validation_anchor,
        "state_window": [(date.fromisoformat(profile.validation_anchor) - timedelta(days=89)).isoformat(), profile.validation_anchor],
        "target_window": [profile.validation_target_start, profile.validation_target_end],
        "root_seed": seed, "cohort_name": cohort_name, "row_count": len(users),
        "ordered_user_id_sha256": _ordered_user_hash(users), "transition_counts": counts,
        "transition_shares": {key: counts[key] / len(users) for key in TRANSITIONS},
        "source_universe_row_count": len(source_users), "source_universe_sha256": source_sha,
        "allocation_method": "largest_remainder", "selection_hash_method": "SHA256(seed|profile|cohort|transition|user_id)",
        "created_by_commit_sha": git_sha(),
    }


def build_nested_cohorts(*, profile: TemporalProfile, train_path: Path, sample_submit_path: Path, output_root: Path, root_seed: int = 42) -> dict[str, Any]:
    template = pl.read_csv(sample_submit_path).select("user_id")
    if template.height != 250_000 or template["user_id"].n_unique() != 250_000:
        raise ValueError("sample_submit.csv must contain 250000 unique user_id values")
    users = template["user_id"].to_list()
    raw = pl.read_parquet(train_path)
    if raw["event_date"].dtype == pl.Utf8:
        raw = raw.with_columns(pl.col("event_date").str.to_date())
    states = build_state_labels(raw, users, profile.validation_anchor).with_columns(
        (pl.col("was_active").cast(pl.Utf8) + pl.col("will_buy").cast(pl.Utf8)).alias("transition")
    ).select("user_id", "transition")
    universe_sha = _ordered_user_hash(users)
    source_counts = {key: states.filter(pl.col("transition") == key).height for key in TRANSITIONS}

    def choose(source: pl.DataFrame, name: str, total: int) -> pl.DataFrame:
        quotas = _largest_remainder(source_counts, total)
        selected = []
        for transition in TRANSITIONS:
            values = source.filter(pl.col("transition") == transition)["user_id"].to_list()
            selected.extend(sorted(values, key=lambda user_id: _rank(profile.name, name, transition, user_id, root_seed))[:quotas[transition]])
        result = template.filter(pl.col("user_id").is_in(selected)).join(states, on="user_id", how="inner")
        observed = {key: result.filter(pl.col("transition") == key).height for key in TRANSITIONS}
        if result.height != total or observed != quotas:
            raise RuntimeError(f"Cohort quota integrity failure for {name}")
        return result

    full = choose(states, "full_cohort_100k", 100_000)
    screen = choose(full, "screen_cohort_25k", 25_000)
    full_users, screen_users = full["user_id"].to_list(), screen["user_id"].to_list()
    if not set(screen_users) < set(full_users) < set(users):
        raise RuntimeError("Nested cohort subset relation failed")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "universe_250k_manifest.json", _manifest(profile=profile, cohort_name="universe_250k", users=users, transitions=states, source_users=users, source_sha=universe_sha, seed=root_seed))
    for name, frame, selected_users in (("full_cohort_100k", full, full_users), ("screen_cohort_25k", screen, screen_users)):
        frame.select("user_id").write_parquet(output_root / f"{name}.parquet")
        write_json(output_root / f"{name}_manifest.json", _manifest(profile=profile, cohort_name=name, users=selected_users, transitions=frame, source_users=users, source_sha=universe_sha, seed=root_seed))
    comparison = {"transition": {key: {"universe": source_counts[key], "full": full.filter(pl.col("transition") == key).height, "screen": screen.filter(pl.col("transition") == key).height} for key in TRANSITIONS}, "train_sha256": sha256_file(train_path), "sample_submit_sha256": sha256_file(sample_submit_path)}
    write_json(output_root / "cohort_comparison.json", comparison)
    return comparison
