# Direct temporal CV v1 — extension runbook

The baseline path is `configs/direct_temporal_cv_v1/baseline_catboost.yaml`.
It is the only config currently allowed to run. ETT, TCN and BTYD are wired
as explicit extension points but their adapters intentionally fail fast until
their parity tests are implemented.

## Contract-check commands

Run from the repository root in `myenv`:

```powershell
$py = 'C:\Users\egorg\anaconda3\envs\myenv\python.exe'
& $py scripts/run_direct_temporal_cv.py --experiment-config configs/direct_temporal_cv_v1/baseline_catboost.yaml --contract-check
& $py scripts/run_direct_temporal_cv.py --experiment-config configs/direct_temporal_cv_v1/ett_direct.yaml --contract-check
& $py scripts/run_direct_temporal_cv.py --experiment-config configs/direct_temporal_cv_v1/tcn_direct.yaml --contract-check
```

These commands are read-only. They validate protocol, model IDs and required
stores; they do not train or submit a DataSphere job.

## Intended experiment order

1. Run and validate the four-fold direct CatBoost baseline on 250k.
2. Fill `DirectBTYDFeatureProvider`, then run `catboost_btyd_pending.yaml`.
3. Fill the scalar-head `DirectETTAdapter`; run ETT as a standalone direct
   model, not as a specialist or hurdle component.
4. Fill `DirectTCNAdapter`; run the same standalone direct target and folds.
5. Save independent prediction banks. Only then fit `CB+ETT`, `CB+TCN` and
   `CB+ETT+TCN` blends using F1--F3; F4 remains an untouched gate.

## Implementation rules

- Every adapter creates a fresh model, optimizer, scheduler and scaler per fold.
- ETT reuses the existing `EventTimeTransformer`/FP16 event store, but has one
  scalar `log1p(GMV30)` head; no SSL, transition heads, H/F specialists or meta.
- TCN reuses causal daily-tensor blocks with one scalar direct head; no SSL or
  transition balancing.
- BTYD fits parameters only through each fold's train anchor and may expose
  exactly `btyd_p_buy_30d`, `btyd_expected_purchases_30d` and `btyd_p_alive`.
- No adapter may read future rows, validation labels, or test labels.
- Use `scripts/datasphere_runner.py` and a new PRE-RUN commit for each enabled
  config. Do not launch pending configs by renaming them or bypassing the
  adapter fail-fast.

## DataSphere manifest template

Create one manifest per experiment only after local contract and smoke tests:

```yaml
name: direct-temporal-cv-ett-v1
cmd: >
  export PYTHONPATH=. ; python3 scripts/run_direct_temporal_cv.py
  --experiment-config configs/direct_temporal_cv_v1/ett_direct.yaml
env:
  python:
    type: manual
    version: 3.10.13
    requirements-file: requirements-datasphere.txt
    local-paths:
      - src/
      - scripts/
      - configs/direct_temporal_cv_v1/
      - data/train.parquet
      - sample_submit.csv
outputs:
  - artifacts/direct_temporal_cv_v1/experiments/direct_cv_ett_v1/
cloud-instance-types:
  - g1.1
```

Use a separate output root and job for every model/feature variant. Never run
ETT/TCN/BTYD together before the single-model fold metrics have been recorded.
