"""Generate immutable one-model individual-screen configs."""

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
            result[key] = copy.deepcopy(value)
    return result


def _changed_paths(before: Any, after: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: set[tuple[str, ...]] = set()
        for key in set(before) | set(after):
            result |= _changed_paths(before.get(key), after.get(key), (*prefix, str(key)))
        return result
    return set() if before == after else {prefix}


def generate_individual_search(plan_path: Path, output_dir: Path) -> list[Path]:
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"search_id", "base_config", "target_model", "stage", "candidates"}:
        raise ValueError("Search plan fields must be search_id, base_config, target_model, stage, candidates")
    if raw["stage"] not in {"screen", "full"}:
        raise ValueError("Individual search supports screen/full stages")
    base = yaml.safe_load(Path(raw["base_config"]).read_text(encoding="utf-8"))
    target = str(raw["target_model"])
    if target not in base.get("enabled_models", []):
        raise ValueError("target_model must be enabled by base config")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in raw["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {"id", "overrides"} or candidate["id"] in seen:
            raise ValueError("Each candidate needs unique id and overrides")
        seen.add(candidate["id"])
        resolved = _merge(base, candidate["overrides"])
        changed = _changed_paths(base.get("models", {}), resolved.get("models", {}))
        if any(path[:1] != (target,) for path in changed):
            raise ValueError(f"Candidate {candidate['id']} changes a non-target model: {sorted(changed)}")
        candidate_id = str(candidate["id"])
        resolved["experiment_id"] = candidate_id
        resolved["stage"] = raw["stage"]
        resolved["output_root"] = f"artifacts/reference_v1/experiments/{candidate_id}"
        resolved["screening"] = {"target_model": target}
        path = output_dir / f"{candidate_id}.yaml"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite generated config: {path}")
        path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        load_experiment_config(path)
        result.append(path)
    return result
