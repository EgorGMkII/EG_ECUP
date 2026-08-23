# Reference framework V1

`reference_framework_v1` is a config-driven wrapper around the frozen
`SSL_TEMPORAL_STACK_V1` implementations. It does not replace the historical
SSL entrypoint or record/Public builders.

## Commands

```text
python scripts/build_reference_cohorts.py
python scripts/run_reference_experiment.py --experiment-config <config> --pre-run-sha <sha>
python scripts/expand_reference_sweep.py --sweep <sweep.yaml> --output-dir <configs-dir>
```

Each experiment YAML enables a canonical ordered subset of `catboost`, `s1`,
`s2`, `ett`, `tcn`, and `residual_mlp`. The generated prediction schema and schema-v2 joint meta use
the enabled models' named output columns. RUN A fits meta only at M; RUN B is
fresh and applies only the frozen RUN A package at V.

`POST_NY_PUBLIC_PROXY` uses M=`2025-12-15`, V=`2026-01-14`, and evaluates the
known target window `2026-01-15 .. 2026-02-13`. Final inference is explicitly
defined at `2026-02-13` after an approved full experiment pins its config and
meta hashes.

## Cohorts

The builder creates 250k universe, 100k full, and 25k screen cohorts under
`artifacts/reference_v1/cohorts/POST_NY_PUBLIC_PROXY/`. Membership is stable
SHA256 ranking per transition, while parquet ordering always follows
`sample_submit.csv`. No transition sampler, row weighting, or loss weighting
is used for base pooled datasets.

## Experiment safety

- Every config has one unique output root and cannot overwrite a prior result.
- `steps` are explicit selected training endpoints; no checkpoint is selected
  automatically from M or V.
- Anchor recency tickets only affect deterministic neural anchor scheduling.
  CatBoost remains a natural pooled dataset.
- TCN is an independent causal 180-day daily-tensor React/Churn adapter.
  Residual MLP is an independent tabular React/Churn/Amount adapter. Both are
  created afresh in RUN A and RUN B; neither consumes prediction columns from
  the existing stack.
- BTYD is a CatBoost-only feature provider. In each run it fits BG/NBD and
  Gamma-Gamma only on allowed train-anchor frames, then produces causal
  frequency, recency, age, monetary, alive-probability and 30-day expected
  purchase/GMV features for CatBoost. It does not read target columns.
- `post_ny_tcn_mlp_btyd_full.yaml` and its DataSphere manifest define the
  first all-in six-model candidate. The SSL parity config remains frozen.
- `experiments/post_ny_ssl_parity_tcn_mlp_btyd_selected_100k.yaml` is the
  launch-ready all-in comparison using the same immutable
  `selected_users_100k.parquet` as the SSL parity run. It intentionally skips
  the 25k screen stage and evaluates the six-model stack directly on the
  established 100k M/V protocol.

The candidate implementations live under `src/reference_framework_v1/candidates/`;
the thin adapters live in the registry and share the framework's exact-step,
deterministic-anchor, dynamic-prediction-bank and frozen-meta contracts.

The next experiment phase is governed by
[`REFERENCE_FRAMEWORK_V1_HYPERPARAMETER_SCREENING.md`](REFERENCE_FRAMEWORK_V1_HYPERPARAMETER_SCREENING.md): sequential 25k screening followed by fresh 100k confirmation, never a Cartesian sweep or Public-LB selection.
