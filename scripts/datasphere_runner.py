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


def launch_job(config_path: str, project_id: str, token: str) -> str:
    print(f"[*] Submitting DataSphere Job for config {config_path} in project {project_id}...", flush=True)
    # ``execute`` is synchronous unless --async is supplied: synchronous mode
    # streams remote logs and waits for completion before returning.  Request its
    # documented execution-data JSON so the existing monitor can receive the ID.
    with tempfile.TemporaryDirectory(prefix="datasphere_runner_") as tmpdir:
        execution_data_path = Path(tmpdir) / "execution.json"
        cmd = [
            DATASPHERE_EXE, "-t", token, "project", "job", "execute",
            "-p", project_id, "-c", config_path,
            "--async", "--output", str(execution_data_path),
        ]
        out = run_command(cmd, token)
        execution_data = execution_data_path.read_text(encoding="utf-8") if execution_data_path.exists() else ""
    print(out, flush=True)

    # Extract Job ID from the documented async execution-data output, with the
    # former CLI-output parsing retained as compatibility fallback.
    try:
        job_id = json.loads(execution_data).get("job_id", "")
    except json.JSONDecodeError:
        job_id = ""
    match = re.search(r"\b(cbt[a-zA-Z0-9_-]+)\b", out)
    job_id = job_id or (match.group(1) if match else "")
    if job_id:
        print(f"[+] Found DataSphere Job ID: {job_id}", flush=True)
        return job_id
    else:
        # Check if job list has the newest job
        print("[-] Could not parse Job ID directly, fetching recent project jobs...", flush=True)
        list_cmd = [DATASPHERE_EXE, "-t", token, "project", "job", "list", "-p", project_id]
        list_out = run_command(list_cmd, token)
        print(list_out, flush=True)
        return ""


def monitor_and_download(job_id: str, project_id: str, token: str, poll_interval: int = 60):
    print(f"\n[*] Monitoring DataSphere Job {job_id}...", flush=True)
    while True:
        # CLI 0.10.0 exposes ``get`` (not ``status``) for a job's current state.
        status_cmd = [DATASPHERE_EXE, "-t", token, "project", "job", "get", "--id", job_id]
        status_out = run_command(status_cmd, token)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status_out.strip()}", flush=True)

        if "COMPLETED" in status_out.upper() or "SUCCESS" in status_out.upper():
            print(f"\n[+] Job {job_id} COMPLETED SUCCESSFULLY! Downloading output files...", flush=True)
            dl_cmd = [DATASPHERE_EXE, "-t", token, "project", "job", "download-files", "--id", job_id]
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

    args = parser.parse_args()
    token = get_token(args.token)

    if args.download_only and args.id:
        dl_cmd = [DATASPHERE_EXE, "-t", token, "project", "job", "download-files", "--id", args.id]
        print(run_command(dl_cmd, token))
        return

    if args.id:
        monitor_and_download(args.id, args.project_id, token)
        return

    job_id = launch_job(args.config, args.project_id, token)
    if job_id:
        monitor_and_download(job_id, args.project_id, token)


if __name__ == "__main__":
    main()
