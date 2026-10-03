# DataSphere runbook

## Required execution path

Do not call `datasphere.exe` directly. Use only the checked-in runner from the
`myenv` interpreter:

```powershell
C:\Users\egorg\anaconda3\envs\myenv\python.exe scripts\datasphere_runner.py -c <manifest>.yaml
```

For an existing job, use the same runner:

```powershell
C:\Users\egorg\anaconda3\envs\myenv\python.exe scripts\datasphere_runner.py --id <job-id>
```

The runner owns CLI selection, token lookup, child-environment proxy cleanup,
asynchronous submission, job monitoring, and output download. Never put a token
in a manifest, command transcript, log, or committed file. Proxy variables are
cleared only from the runner's child CLI environment, not from the user's shell.

## Before a run

Record the current commit with `git rev-parse HEAD`, then record it beside the
job ID. Review the manifest's `local-paths`, entrypoint, `outputs`, and requested
hardware. Do not infer an unsupported CPU instance type; use a known manifest
format or obtain an explicit approved type.

For the verified smoke probe, the only payload was `scripts/datasphere_smoke.py`
and `requirements-datasphere.txt` (1,077 B total), with `g1.1`, entrypoint
`python3 datasphere_smoke.py`, and output `smoke_result.json`. No raw data,
models, checkpoints, training, or inference belonged in that run.

## Runner/CLI contract

With DataSphere CLI 0.10.0, `project job execute` is synchronous by default.
The runner must use `--async --output <temporary JSON>` and take `job_id` from
that JSON. Poll with `project job get --id`, whose terminal success status is
`SUCCESS`, then use the runner's `download-files` branch. Do not use `attach` as
a read-only log command: this CLI implementation calls `execute` again.

The runner's sanitized console output provides submission, job-state, and
download diagnostics. Store only sanitized excerpts; never store tokens. Link
every job to its pre-run commit. The successful smoke reference is job
`bt1g4pnfonpcud9tfj7m` at commit
`aa3364f45e1f8a51dc27d4cc325bb443c96e057a`.
