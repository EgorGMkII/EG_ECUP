"""Deterministic expansion of one sweep into one config per candidate."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .config import load_experiment_config


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_sweep(path: Path, output_dir: Path) -> list[Path]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(raw) != {"sweep_id", "base_config", "candidates"}:
        raise ValueError("Sweep must contain sweep_id, base_config, candidates only")
    base = yaml.safe_load(Path(raw["base_config"]).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in raw["candidates"]:
        if set(candidate) != {"id", "overrides"} or candidate["id"] in seen:
            raise ValueError("Candidates require unique id and overrides")
        seen.add(candidate["id"])
        resolved = _merge(base, candidate["overrides"])
        resolved["experiment_id"] = candidate["id"]
        resolved["output_root"] = f"artifacts/reference_v1/experiments/{candidate['id']}"
        target = output_dir / f"{candidate['id']}.yaml"
        target.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        load_experiment_config(target)
        paths.append(target)
    return paths
