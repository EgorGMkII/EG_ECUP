"""Deterministic runtime, logging, and manifest helpers for SSL V1."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch


def derive_seed(root_seed: int, *parts: object) -> int:
    payload = "|".join([str(root_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def seed_everything(root_seed: int, *parts: object) -> int:
    seed = derive_seed(root_seed, *parts)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("SSL_TEMPORAL_STACK_V1 requires CUDA; CPU fallback is forbidden")
    return torch.device("cuda:0")


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def progress(event: str, **values: Any) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
