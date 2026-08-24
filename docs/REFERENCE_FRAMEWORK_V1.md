# Reference framework V1

`reference_framework_v1` is a config-driven wrapper around the frozen
`SSL_TEMPORAL_STACK_V1` implementations. It does not replace the historical
SSL entrypoint or record/Public builders.

## Commands

```text
python scripts/build_reference_cohorts.py
python scripts/run_reference_experiment.py --experiment-config <config> --pre-run-sha <sha>
python scripts/expand_reference_sweep.py --sweep <sweep.yaml> --output-dir <configs-dir>
python scripts/generate_reference_search.py --plan <search.yaml> --output-dir <configs-dir>
python scripts/run_reference_individual_screen.py --experiment-config <config> --output-root <root> --pre-run-sha <sha>
python scripts/run_reference_incremental_gate.py --experiment-config <config> --baseline-root <root> --candidate-root <root> --output-root <root> --pre-run-sha <sha>
python scripts/prepare_reference_screen_baseline.py --base-config <config> --experiment-id <id> --output-config <config>
```

Each experiment YAML enables a canonical ordered subset of `catboost`,
`catboost_direct`, `s1`, `s2`, `ett`, `tcn`, and `residual_mlp`. The generated prediction schema and named-column joint meta use
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
- BTYD follows the accepted `B1_BTYD_ProbCount_ClassifierOnly` contract from
  `BTYD_LEAKAGE_AUDIT.md`. In each run it builds exact causal full-history RFM
  from raw events, fits BG/NBD on at most 50k purchasing rows from training
  anchors only, and exposes exactly `btyd_p_buy_30d`,
  `btyd_expected_purchases_30d`, and `btyd_p_alive` to CatBoost React/Churn.
  The sampling is deterministic from the run seed. Amount receives no BTYD
  feature; Gamma-Gamma is not fitted. Future labels are never feature inputs.
- `catboost_direct` is a separate unconditional regressor. It uses sparse
  7/14/30/60/90/180/365-day monetary, frequency, recency, and history
  aggregates, trains on `log1p(next_30d_gmv)` at the latest causal training
  anchor, and emits `cb_direct_z`. It is not an Amount specialist. Meta schema
  v3 first constructs the complete hurdle prediction and then fits a simplex
  late blend of `[hurdle_prediction_z, cb_direct_z, ...]` on RUN A/M. RUN B
  applies those frozen late-blend weights. Schema-v2 packages without direct
  models retain implicit `hurdle=1` behavior.
- `post_ny_tcn_mlp_btyd_full.yaml` and its DataSphere manifest define the
  first all-in six-model candidate. The SSL parity config remains frozen.
- `experiments/post_ny_ssl_parity_tcn_mlp_btyd_selected_100k.yaml` is the
  launch-ready all-in comparison using the same immutable
  `selected_users_100k.parquet` as the SSL parity run. It intentionally skips
  the 25k screen stage and evaluates the six-model stack directly on the
  established 100k M/V protocol.

The candidate implementations live under `src/reference_framework_v1/candidates/`;
the thin adapters live in the registry and share the framework's deterministic
epoch/step resolution, dynamic-prediction-bank and frozen-meta contracts.

The next experiment phase is governed by
[`REFERENCE_FRAMEWORK_V1_HYPERPARAMETER_SCREENING.md`](REFERENCE_FRAMEWORK_V1_HYPERPARAMETER_SCREENING.md): sequential 25k screening followed by fresh 100k confirmation, never a Cartesian sweep or Public-LB selection.

## Cohort scaling and optimizer budgets

Raw optimizer-step counts are not scale invariant. Epoch recipes are the
active framework contract: one epoch exhausts every anchor loader exactly once
before any anchor restarts. With 14 anchors and a batch of 512, base training
needs 686 steps per epoch at 25k, 2,744 at 100k, and 6,846 at 250k. ETT counts
micro-batches and accumulation explicitly; specialist epoch sizes use the
actual React, Churn, or Amount subset rather than the full frame.

Use nested successive halving instead of copying an old step count:

1. screen many recipes on 25k;
2. rerun only the best one or two from scratch on 100k;
3. run a fresh 250k validation gate for the confirmed protocol;
4. train the final 250k protocol with its fixed epoch recipe and record
   the resolved steps, examples and per-anchor batches in the training report.

The default candidate family is two or three base epochs, one SSL epoch where
applicable, and one H plus one F epoch per specialist. LR, dropout and
relative loss weights may transfer from screen/full; scheduler duration and
warmup are resolved from the final epoch count. A larger cohort alone is not a
reason to lower peak LR. Lower it only when the longer schedule shows unstable
updates or repeated-pass overfit.

## Hyperparameter selection workflow

`configs/reference_framework_v1/searches/` declares one-model searches. A
generated individual screen trains only `screening.target_model` in independent
RUN A and RUN B and reports its own React/Churn/Amount metrics; it does not
load or retrain other first-level models. The first search order is hurdle
CatBoost specialists, ETT, then TCN. S1/S2, Residual MLP and direct CatBoost
remain fixed during this cycle. Direct CatBoost is excluded from tuning banks
(equivalent to a fixed late-blend weight of zero) and is introduced only after
the neural/CatBoost-specialist recipe is selected.

Only manually selected top individual candidates enter an incremental gate.
The gate replaces their named prediction columns in immutable baseline M/V
banks, validates ordered identity columns and SHA, refits meta on M, and
scores V. `collect_reference_search_results.py` writes a unified leaderboard;
`compare_reference_gates.py` performs paired bootstrap on aligned V banks.
`promote_reference_candidate.py` creates the next immutable config but never
launches it. Promotion is screen -> fresh 100k -> fresh 250k validation ->
final 250k.
