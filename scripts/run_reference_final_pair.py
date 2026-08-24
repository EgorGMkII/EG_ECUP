"""Run the two pinned 250k final jobs sequentially with live DataSphere logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import TextIO

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "datasphere_runner.py"
VERIFIER = ROOT / "scripts" / "verify_reference_experiment_artifacts.py"

JOBS = (
    (
        "no_direct",
        ROOT / "datasphere.reference_framework_v1_final_six_model_no_direct_v1.yaml",
        ROOT / "artifacts/reference_v1/experiments/post_ny_final_six_model_no_direct_v1",
    ),
    (
        "with_direct",
        ROOT / "datasphere.reference_framework_v1_final_six_model_with_direct_v1.yaml",
        ROOT / "artifacts/reference_v1/experiments/post_ny_final_six_model_with_direct_v1",
    ),
)


def _require_myenv() -> None:
    if Path(sys.executable).resolve().parent.name.lower() != "myenv":
        raise RuntimeError(f"Run with the myenv interpreter, got: {sys.executable}")


def _manifest_stage(path: Path) -> str:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    command = str(manifest.get("cmd", ""))
    marker = "--experiment-config "
    if marker not in command:
        raise ValueError(f"Manifest has no experiment config: {path}")
    config_path = ROOT / command.split(marker, 1)[1].split()[0].strip("'\"")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(config.get("stage", ""))


def _stream(command: list[str], transcript: TextIO) -> None:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        transcript.write(line)
        transcript.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _assert_clean_destination(root: Path) -> None:
    protected = ("final_manifest.json", "submission.csv", "artifact_sha256.json")
    existing = [str(root / name) for name in protected if (root / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite/download over final artifacts: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run-sha", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _require_myenv()
    if len(args.pre_run_sha) < 7:
        raise ValueError("--pre-run-sha must identify the pinned PRE-RUN commit")

    for _name, manifest, output_root in JOBS:
        if not manifest.is_file() or _manifest_stage(manifest) != "final":
            raise ValueError(f"Not a valid final manifest: {manifest}")
        if not args.dry_run:
            _assert_clean_destination(output_root)

    commands = [
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(manifest),
            "--pre-run-sha",
            args.pre_run_sha,
            "--sync-submit",
        ]
        for _name, manifest, _root in JOBS
    ]
    if args.dry_run:
        print(json.dumps({"mode": "sequential_sync", "commands": commands}, indent=2), flush=True)
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_root = ROOT / "artifacts/reference_v1/final_pair_runs" / run_id
    log_root.mkdir(parents=True, exist_ok=False)
    for (name, _manifest, output_root), command in zip(JOBS, commands, strict=True):
        transcript_path = log_root / f"{name}.log"
        print(f"[*] Starting {name}; transcript={transcript_path}", flush=True)
        with transcript_path.open("w", encoding="utf-8") as transcript:
            _stream(command, transcript)
        verify_command = [
            sys.executable,
            str(VERIFIER),
            "--root",
            str(output_root),
            "--expected-rows",
            "250000",
        ]
        subprocess.run(verify_command, cwd=ROOT, check=True)
        print(f"[+] Verified {name}; proceeding to the next final job", flush=True)


if __name__ == "__main__":
    main()
