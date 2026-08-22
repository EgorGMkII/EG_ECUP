"""Autonomous DataSphere Job Runner for Yandex DataSphere GPU execution."""

import argparse
import os
import re
import subprocess
import sys
import time

DEFAULT_PROJECT_ID = "bt1pnckp8jvj2ckm20mu"
DATASPHERE_EXE = r"C:\Users\egorg\anaconda3\envs\myenv\Scripts\datasphere.exe"


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


def launch_job(config_path: str, project_id: str, token: str) -> str:
    print(f"[*] Submitting DataSphere Job for config {config_path} in project {project_id}...", flush=True)
    # Synchronous execute is intentional: DataSphere CLI streams remote stdout,
    # stderr, docker stats and GPU stats into its local per-job log directory.
    cmd = [
        DATASPHERE_EXE, "-t", token, "project", "job", "execute",
        "-p", project_id, "-c", config_path,
    ]
    try:
        out = stream_command(cmd, token)
    except subprocess.CalledProcessError as error:
        out = error.output or ""
        print(f"[-] Synchronous DataSphere execution exited with code {error.returncode}.", flush=True)

    job_id = extract_job_id(out)
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
