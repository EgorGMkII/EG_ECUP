"""Minimal DataSphere environment smoke test; intentionally contains no ML work."""

import json
import platform
import sys
from pathlib import Path

import numpy  # noqa: F401
import pandas  # noqa: F401
import polars  # noqa: F401
import torch


def main() -> None:
    result = {
        "status": "OK",
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    print(f"Python: {result['python_version']}")
    print("Imports: numpy, pandas, polars, torch")
    print(f"torch.cuda.is_available(): {result['torch_cuda_available']}")
    Path("smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("STATUS: OK")


if __name__ == "__main__":
    main()
