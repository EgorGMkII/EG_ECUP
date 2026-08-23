# SSL_TEMPORAL_STACK_V1 implementation status

This file records engineering gates only. None of the smoke scores below is a
baseline or a validation result.

## Current state

- Frozen specification: `docs/SSL_TEMPORAL_STACK_V1_SPEC.md`.
- Full CLI: `scripts/run_ssl_temporal_stack_v1.py`.
- Full DataSphere manifest: `datasphere.ssl_temporal_stack_v1_full.yaml`.
- GPU smoke manifest: `datasphere.ssl_temporal_stack_v1_smoke.yaml`.
- Local tests: 37 passed on 2026-08-23 in conda environment `myenv`.
- Full DataSphere job: not launched.
- GPU DataSphere smoke: not launched.
- PRE-RUN commit: not created yet.

## Local 100-user smoke

Successful isolated run:

```text
artifacts/ssl_temporal_stack_v1/local_smoke_20260823_v3
status: COMPLETED
elapsed: 13.0192614 seconds
train anchor: 2025-11-10
holdout anchor: 2025-12-15
cohort: 25 users from each training transition 00/01/10/11
CatBoost: CPU, 2 iterations per task
S1/S2/ETT: one optimizer step per SSL/base/H/F phase
```

The reported RMSLE `1.6912552118701238` is an in-sample M meta-fit smoke value
after one-step models. It has no experimental meaning and must not be compared
with either Public LB or temporal validation.

Artifact SHA256:

```text
smoke_manifest.json
c6eb1477404fb02d839fa24308462bd30f9145b641917bbc90a9b204027ba96b

smoke_prediction_bank.parquet
5c5dbf5899a9a0ef908e5547ae7902e25d39a903768e7494c7d1f8db29db7c33

smoke_meta_package.json
caa2ac6a67b05bfe48f40099ec4ac83c8136ca0360be7b2beea80c9a4549bf52

smoke_report.json
d5d290f6b75a186cdfc8afdd8abe878c88e52081c2c5007ed9c0a702c6b2f47c
```

## Defects found by the local smoke

1. `ts_gmv_skewness_90d` and `ts_gmv_kurtosis_90d` legitimately contain NaN
   for insufficient-variance histories. The contract now permits NaN only in
   these two columns and relies on CatBoost `nan_mode="Min"`; every infinity or
   NaN in another feature remains fatal.
2. The historical ETT extraction function filtered by date but not by the
   requested cohort before creating its Python user dictionary. The new store
   filters raw events to the run cohort once before processing anchors. This
   removed the smoke OOM/native termination and reduces full-run memory.

Failed smoke directories `local_smoke_20260823_v1` and
`local_smoke_20260823_v2` are forensic failure records and are not results.

## Remaining gates before a full job

1. Compile, unit tests, YAML parsing, manifest path validation.
2. Create a PRE-RUN commit containing code, configs, manifests, tests, and docs.
3. Launch only `datasphere.ssl_temporal_stack_v1_smoke.yaml` through
   `scripts/datasphere_runner.py` in `myenv`.
4. Verify real CUDA, CatBoost GPU, AMP, FP16 event memory maps, exact step counts,
   five downloaded smoke outputs, and live logs.
5. Only after a successful smoke and explicit authorization, launch the full
   manifest from the same PRE-RUN SHA.
