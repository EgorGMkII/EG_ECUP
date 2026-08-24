"""Run-scoped, non-overwriting artifact and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_root(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite direct-CV output root: {path}")
    path.mkdir(parents=True)
    return path


def write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
