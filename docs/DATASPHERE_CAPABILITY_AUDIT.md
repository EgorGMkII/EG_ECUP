# DataSphere capability audit

## Status: `DATASPHERE_EXECUTION_VERIFIED`

The verified execution path is exclusively the repository runner:

```powershell
C:\Users\egorg\anaconda3\envs\myenv\python.exe scripts\datasphere_runner.py -c datasphere.smoke.yaml
```

The runner uses `C:\Users\egorg\anaconda3\envs\myenv\Scripts\datasphere.exe`
and project `bt1pnckp8jvj2ckm20mu`. It sources `YC_TOKEN` from an explicit
argument, environment, or the Windows user registry, without printing or saving
the value. Each CLI child receives `YC_CLI_INITIALIZATION_SILENCE=true`; only the
child environment has `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and lowercase
equivalents removed.

## Verified smoke job

- pre-run Git commit: `aa3364f45e1f8a51dc27d4cc325bb443c96e057a`;
- successful job ID: `bt1g4pnfonpcud9tfj7m`;
- requested hardware: `g1.1` (the existing manifest format; no CPU type was
  guessed);
- payload: `scripts/datasphere_smoke.py` (864 B) and
  `requirements-datasphere.txt` (213 B), total 1,077 B; no data, models,
  checkpoints, or other artifacts;
- entrypoint: `python3 datasphere_smoke.py`;
- declared and downloaded output: `smoke_result.json` (224 B).

The job reached `PREPARING`, `EXECUTING`, `UPLOADING_OUTPUT`, then `SUCCESS`.
The downloaded result records `status: OK`, Python 3.10.12, Torch 2.3.1+cu121,
and `torch_cuda_available: true`.

Sanitized runner output confirms job creation, state transitions, and one output
file downloaded. It contains no token. The installed CLI emits a local
`RequestsDependencyWarning` concerning its own `urllib3`/`charset_normalizer`
versions, but the remote job completed successfully.

## Runner compatibility corrections

The existing runner was retained and minimally corrected for installed
DataSphere CLI 0.10.0:

1. `execute` is invoked with documented `--async --output <temporary JSON>` so
   its job ID is available immediately instead of waiting synchronously for
   remote completion.
2. Status polling uses supported `project job get --id`; CLI 0.10.0 has no
   `project job status` command.
3. `SUCCESS` is accepted as a successful terminal state and triggers the
   existing `download-files` step.

An earlier smoke job, `bt1mie94c4jdc1plqn9t`, ended in `ERROR`: its entrypoint
used `scripts/datasphere_smoke.py`, while a single-file `local-paths` archive is
unpacked at the job root. The corrected entrypoint above was then verified.
