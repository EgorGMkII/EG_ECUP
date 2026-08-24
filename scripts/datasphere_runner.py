"""Autonomous DataSphere Job Runner for Yandex DataSphere GPU execution."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_PROJECT_ID = "bt1pnckp8jvj2ckm20mu"
DATASPHERE_EXE = r"C:\Users\egorg\anaconda3\envs\myenv\Scripts\datasphere.exe"
DATASPHERE_PYTHON = r"C:\Users\egorg\anaconda3\envs\myenv\python.exe"
DATASPHERE_WRAPPER = Path(__file__).with_name("datasphere_cli_wrapper.py")


def datasphere_command(*args: str) -> list[str]:
    """Build a CLI invocation with the local no-network version-check shim."""
    return [DATASPHERE_PYTHON, str(DATASPHERE_WRAPPER), *args]


def extract_job_id(output: str) -> str:
    """Extract only an explicitly labelled job ID, never an operation/project ID."""
    patterns = (
        r"created job\s+[`'\"]((?:bt1|cbt)[a-zA-Z0-9_-]+)[`'\"]",
        r"job[_ -]?id\s*[:=]\s*[`'\"]?((?:bt1|cbt)[a-zA-Z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def get_token(args_token: str = None) -> str:
    if args_token:
        return args_token
    if "YC_TOKEN" in os.environ and os.environ["YC_TOKEN"]:
        return os.environ["YC_TOKEN"]
    # Fallback to Windows User Registry
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        tok, _ = winreg.QueryValueEx(key, "YC_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    raise ValueError("YC_TOKEN not found! Please set $env:YC_TOKEN or pass via --token.")


def run_command(cmd_list: list, token: str = None) -> str:
    env = os.environ.copy()
    if token:
        env["YC_TOKEN"] = token
    env["YC_CLI_INITIALIZATION_SILENCE"] = "true"
    # Prevent WinError 10061 if rogue proxy is present in shell
    for p in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(p, None)

    res = subprocess.run(cmd_list, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    return res.stdout


def stream_command(cmd_list: list, token: str = None) -> str:
    """Run a synchronous CLI command while forwarding every output line."""
    env = os.environ.copy()
    if token:
        env["YC_TOKEN"] = token
    env["YC_CLI_INITIALIZATION_SILENCE"] = "true"
    for proxy_name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(proxy_name, None)

    process = subprocess.Popen(
        cmd_list,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd_list, "".join(output))
    return "".join(output)


def resolve_pre_run_sha(value: str = None) -> str:
    """Return the explicit SHA or the current tracked commit for a job manifest."""
    if value:
        return value
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout.strip()


def materialize_runtime_config(config_path: str, pre_run_sha: str = None) -> tuple[str, Path | None]:
    """Fill the tracked PRE-RUN placeholder in a temporary sibling YAML.

    `/job` is not a Git checkout.  Keeping the placeholder in the committed
    manifest and materializing only its job-local value makes the executed SHA
    explicit without modifying the tracked config after its PRE-RUN commit.
    """
    path = Path(config_path)
    if not path.is_file():
        return config_path, None
    source = path.read_text(encoding="utf-8")
    marker = "__PRE_RUN_SHA__"
    if marker not in source:
        return config_path, None
    resolved = resolve_pre_run_sha(pre_run_sha)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=path.suffix,
        prefix=f".{path.stem}.runtime_",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(source.replace(marker, resolved))
        return stream.name, Path(stream.name)


def launch_job(
    config_path: str,
    project_id: str,
    token: str,
    pre_run_sha: str = None,
    async_submit: bool = False,
) -> str:
    print(f"[*] Submitting DataSphere Job for config {config_path} in project {project_id}...", flush=True)
    # Synchronous execute is intentional: DataSphere CLI streams remote stdout,
    # stderr, docker stats and GPU stats into its local per-job log directory.
    effective_config, temporary_config = materialize_runtime_config(config_path, pre_run_sha)
    if temporary_config:
        print(f"[*] Injected PRE-RUN SHA into temporary job config: {temporary_config.name}", flush=True)
    cmd = datasphere_command("-t", token, "project", "job", "execute", "-p", project_id, "-c", effective_config)
    async_output: Path | None = None
    if async_submit:
        # Synchronous execute may stall in the CLI's operation polling before
        # returning a job ID. Async creates the same job; monitoring is then
        # performed explicitly via the safe --id path below.
        cmd.append("--async")
        output_fd, output_name = tempfile.mkstemp(prefix="datasphere_execute_", suffix=".json")
        os.close(output_fd)
        async_output = Path(output_name)
        cmd.extend(["--output", str(async_output)])
    try:
        try:
            out = stream_command(cmd, token)
        except subprocess.CalledProcessError as error:
            out = error.output or ""
            print(f"[-] Synchronous DataSphere execution exited with code {error.returncode}.", flush=True)
    finally:
        if temporary_config and temporary_config.exists():
            temporary_config.unlink()

    job_id = extract_job_id(out)
    if not job_id and async_output and async_output.exists():
        try:
            payload = json.loads(async_output.read_text(encoding="utf-8"))
            serialized = json.dumps(payload, ensure_ascii=False)
            job_id = extract_job_id(serialized)
            if not job_id:
                for key in ("job_id", "jobId", "id"):
                    value = payload.get(key) if isinstance(payload, dict) else None
                    if isinstance(value, str) and re.fullmatch(r"(?:bt1|cbt)[a-zA-Z0-9_-]+", value):
                        job_id = value
                        break
        except (OSError, json.JSONDecodeError):
            pass
    if async_output and async_output.exists():
        try:
            async_output.unlink()
        except PermissionError:
            # A failed CLI child can briefly retain the output handle.
            pass
    if job_id:
        print(f"[+] Found DataSphere Job ID: {job_id}", flush=True)
        return job_id
    else:
        # Check if job list has the newest job
        print("[-] Could not parse Job ID directly, fetching recent project jobs...", flush=True)
        list_cmd = datasphere_command("-t", token, "project", "job", "list", "-p", project_id)
        list_out = run_command(list_cmd, token)
        print(list_out, flush=True)
        return ""


def monitor_and_download(job_id: str, project_id: str, token: str, poll_interval: int = 60):
    print(f"\n[*] Monitoring DataSphere Job {job_id}...", flush=True)
    while True:
        # CLI 0.10.0 exposes ``get`` (not ``status``) for a job's current state.
        status_cmd = datasphere_command("-t", token, "project", "job", "get", "--id", job_id)
        status_out = run_command(status_cmd, token)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status_out.strip()}", flush=True)

        if "COMPLETED" in status_out.upper() or "SUCCESS" in status_out.upper():
            print(f"\n[+] Job {job_id} COMPLETED SUCCESSFULLY! Downloading output files...", flush=True)
            dl_cmd = datasphere_command("-t", token, "project", "job", "download-files", "--id", job_id)
            dl_out = run_command(dl_cmd, token)
            print(dl_out, flush=True)
            print("[+] All artifacts downloaded to local workspace!", flush=True)
            break
        elif "FAILED" in status_out.upper() or "CANCELLED" in status_out.upper() or "ERROR" in status_out.upper():
            print(f"\n[-] Job {job_id} ended with status: {status_out}", flush=True)
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Autonomous DataSphere Job Runner")
    parser.add_argument("-c", "--config", default="datasphere.gru_sweep.yaml", help="Path to DataSphere job YAML config")
    parser.add_argument("-p", "--project-id", default=DEFAULT_PROJECT_ID, help="DataSphere Project ID")
    parser.add_argument("-t", "--token", default=None, help="Yandex Cloud OAuth Token")
    parser.add_argument("--id", default=None, help="Monitor existing Job ID")
    parser.add_argument("--download-only", action="store_true", help="Download files for Job ID")
    parser.add_argument("--pre-run-sha", default=None, help="SHA injected into __PRE_RUN_SHA__ in the runtime-only YAML copy")
    parser.add_argument(
        "--async-submit",
        dest="async_submit",
        action="store_true",
        default=False,
        help="Submit with DataSphere --async, then monitor by returned job ID (opt-in; no live execute stream)",
    )
    parser.add_argument(
        "--sync-submit",
        dest="async_submit",
        action="store_false",
        help="Use blocking synchronous execute and preserve the local live stream (default)",
    )

    args = parser.parse_args()
    token = get_token(args.token)

    if args.download_only and args.id:
        dl_cmd = datasphere_command("-t", token, "project", "job", "download-files", "--id", args.id)
        print(run_command(dl_cmd, token))
        return

    if args.id:
        monitor_and_download(args.id, args.project_id, token)
        return

    job_id = launch_job(args.config, args.project_id, token, args.pre_run_sha, args.async_submit)
    if job_id:
        monitor_and_download(job_id, args.project_id, token)


if __name__ == "__main__":
    main()
