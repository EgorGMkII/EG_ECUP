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
`s2`, and `ett`. The generated prediction schema and schema-v2 joint meta use
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
- TCN, Residual MLP and BTYD are future adapters/feature sets; they are not
  part of V1.

The repository includes non-registered, shape-tested skeletons for all three
under `src/reference_framework_v1/candidates/`. They cannot be enabled through
`enabled_models` until dedicated adapters, recipes, and temporal experiments
are added.
