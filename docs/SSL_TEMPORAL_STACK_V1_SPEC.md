# SSL_TEMPORAL_STACK_V1 — frozen experiment specification

Status: specification only. No implementation, PRE-RUN commit, or DataSphere job
exists for this experiment yet.

This is a new temporal-validation experiment. It is not the reproduced Public
builder that scored `1.663815381013448`, and it must never be reported as a
reproduction of that builder. The Public result remains the immutable external
baseline. This experiment adds explicit SSL and specialist fine-tuning and is
compared with that baseline only after its own validation result is complete.

## 1. Experiment identity and invariants

- Experiment ID: `SSL_TEMPORAL_STACK_V1`.
- Root seed: `42`.
- Cohort: the immutable ordered 100k cohort from
  `selected_users_100k.parquet`.
- Cohort SHA256:
  `d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2`.
- Models: CatBoost, S1 Masked GRU, S2 Multi-Horizon GRU, ETT.
- RUN A and RUN B use identical classes, inputs, losses, batch sizes, optimizer
  settings, step budgets, prediction columns, and specialist scheme.
- RUN B starts from new random initialization. No model, encoder, scaler,
  optimizer, or scheduler state is transferred from RUN A to RUN B.
- The only learned object transferred from RUN A to RUN B is the frozen joint
  meta package.
- All neural budgets are exact optimizer-step budgets. Epoch-based stopping is
  forbidden.
- A phase fails unless `completed_optimizer_steps == requested_optimizer_steps`.
- No per-anchor model fitting and no temporal OOF model loop are performed.
- CUDA is mandatory. CatBoost must report `task_type="GPU"`; PyTorch must fail
  before store construction if CUDA is unavailable.

## 2. Temporal protocol and anchors

All users are retained. Training size is reduced by dropping the oldest anchors,
not by sampling users. This preserves rare reactivation and churn transitions.

### RUN A — fit first-level models and meta package

RUN A uses the last 11 eligible training anchors:

```text
2025-06-23
2025-07-07
2025-07-21
2025-08-04
2025-08-18
2025-09-01
2025-09-15
2025-09-29
2025-10-13
2025-10-27
2025-11-10
```

- Nominal pooled rows: `11 * 100000 = 1,100,000` before task masks.
- Meta anchor M: `2025-12-15`.
- Meta target: `2025-12-16 .. 2026-01-14`, inclusive.
- No label after M is used to train any RUN A first-level model.
- M labels are used only by the joint meta optimizer and diagnostics.

### RUN B — fresh first-level training and frozen-meta validation

RUN B uses the last 14 eligible training anchors:

```text
2025-06-23
2025-07-07
2025-07-21
2025-08-04
2025-08-18
2025-09-01
2025-09-15
2025-09-29
2025-10-13
2025-10-27
2025-11-10
2025-11-24
2025-12-08
2025-12-15
```

- Nominal pooled rows: `14 * 100000 = 1,400,000` before task masks.
- Validation anchor V: `2026-01-14`.
- Validation target: `2026-01-15 .. 2026-02-13`, inclusive.
- The target of training anchor `2025-12-15` ends exactly on V. Equality is
  allowed; a training target ending after V is forbidden.
- V labels are used only after all RUN B predictions and the frozen-meta output
  have been written.

The RUN A anchors are an exact subset of the RUN B anchors. RUN B extends RUN A
with `2025-11-24`, `2025-12-08`, and `2025-12-15`.

## 3. Shared stores and sampling

Stores are built once on the DataSphere VM from raw data for the union of the 11
RUN A anchors, three later RUN B anchors, M, and V. Because M is also a RUN B
training anchor, this is 15 unique anchors in total.

- Feature frames: one parquet per unique anchor, ordered by the immutable cohort.
- Dense daily store: `[100000, 180, 15]` per anchor.
- Event-time store: FP16 memory maps, not NPZ and not all-resident RAM.
- Horizon labels for S2: built once for each training anchor and cached.
- A store for an anchor is never rebuilt inside an epoch or optimizer-step loop.
- Raw data is released after all requested stores and labels are complete.

Every exact-step neural loader samples anchors round-robin. Each anchor owns a
deterministically shuffled user iterator. When an iterator is exhausted it is
recreated with the next derived seed. Therefore a step budget is distributed
across every anchor and can never stop after consuming only the earliest anchors.

The anchor order is the chronological order shown above. Seeds are derived from:

```text
SHA256(root_seed, run, model_id, stage, task, phase, anchor, iterator_cycle)
```

## 4. Input tensors and targets

### CatBoost tabular input

- Input: exactly 374 ordered float32 features. Their machine-readable canonical
  order is checked into
  `configs/ssl_temporal_stack_v1/catboost_feature_order.json`.
- Feature-contract version: `snapshot_374_v1`.
- SHA256 of the compact JSON ordered-name list:
  `e54a385cd69665f09063762a2e8581fe953b759004a69180487dfcea3fa8df6e`.
- The runtime feature-store manifest must match the checked-in list by name,
  position, count, and SHA256 before any CatBoost model starts. A RUN A/RUN B
  mismatch is fatal.
- Builder contract: `src.snapshots.build_snapshot` with `history_days=90` and
  the shared `compute_global_platform_table(raw)` calculated once from raw data.
  The builder may be refactored, but its resulting ordered feature contract and
  numerical parity must remain unchanged for this experiment ID.
- Every feature uses events from `[anchor - 89 days, anchor]`, except global and
  calendar features whose causal cutoff is defined by `src.features`; no event
  after the anchor is allowed.
- Excluded columns: identifiers, anchor date, all future targets, `will_buy`,
  `was_active`, and diagnostic-only columns.
- Recency fields and `customer_age_days` use `999.0` when missing. Other missing
  snapshot aggregates use `0.0`, except `ts_gmv_skewness_90d` and
  `ts_gmv_kurtosis_90d`: these may remain NaN when a user's history has
  insufficient variance, matching the verified Public builder. CatBoost handles
  these two columns with `nan_mode="Min"`. NaN in any other feature and every
  infinity are fatal.
- React training rows: `was_active == 0`; target `will_buy`.
- Churn training rows: `was_active == 1`; target `1 - will_buy`.
- Amount training rows: `future_gmv_30d > 0`; target
  `z_target = log1p(future_gmv_30d)`.

The 374 features are partitioned exactly as follows:

| Family | Count | Definition |
|---|---:|---|
| Window aggregates | 220 | 44 features for each of 7, 14, 30, 60, and 90 days |
| Recency/lifetime | 8 | customer age, four recencies, ever ordered, lifetime orders, lifetime purchase days |
| Time-series dynamics | 9 | spikes, conversion, acceleration, weekend bias, peak/mean, volatility, skewness, kurtosis |
| Global platform | 5 | DAU and global GMV/buyer-rate context |
| Calendar/target-window | 15 | anchor/target calendar, cyclic DOY, holiday overlap and distance |
| Derived rates/ratios | 112 | 14 derived values for each of eight behavioural families |
| Final user/global ratios | 4 | order interval, recency ratio, user/global GMV and activity ratios |
| Available history | 1 | `available_history_days` |

For each window, the 44 aggregate features are:

- four day counts: active, purchase, search, and cart days;
- for each of `gmv`, `to_ord`, `to_cart`, `searches`, `search_to_cart`,
  `search_to_ord`, `cat_to_cart`, and `cat_to_ord`: sum, calendar-day mean,
  active-day mean, maximum, and calendar-day standard deviation.

The 112 derived features cover `gmv`, `to_ord`, `to_cart`, `searches`,
`active_days`, `purchase_days`, `search_days`, and `cart_days`. Each family has
five window rates, three rate differences, three cross-window ratios, and three
zero-denominator indicators. The checked-in JSON, not this prose summary, is the
authority for exact names and order.

#### BTYD policy

BTYD features are intentionally absent from `SSL_TEMPORAL_STACK_V1`.

- The verified 374-feature contract contains zero `btyd_*` columns.
- The successful Public builder `scripts/build_joint_rmsle_submission.py` does
  not import or join either `src.btyd_pipeline` or
  `src.btyd_research_pipeline`.
- Existing BTYD code belongs to separate research experiments and is not part of
  the reproduced Public stack.
- Therefore this experiment means **the verified 374-feature CatBoost contract
  plus neural SSL**, not CatBoost plus BTYD plus SSL.

Adding BTYD would change both the CatBoost feature space and the causal fitting
protocol. It requires a separate experiment ID and specification, for example
`SSL_TEMPORAL_STACK_V1_BTYD`, and cannot be enabled by a runtime flag in this
experiment. Such a specification must separately freeze the BTYD implementation,
fit scope, penalizers, per-anchor reuse policy, unavailable-user values, exact
columns, and feature-order hash.

### S1 and S2 dense daily input

- Tensor shape: `[B, 180, 15]`.
- Time range: anchor minus 179 days through the anchor, inclusive.
- Channel order is frozen:

```text
searches, to_cart, to_ord, gmv,
search_to_cart, search_to_ord, cat_to_cart, cat_to_ord,
gmv_search, gmv_cat,
is_active, is_purchase_day,
sin_dow, cos_dow, normalized_position
```

- The first ten numeric channels are `log1p(max(value, 0))`.
- Binary/calendar/position channels are not standardized.
- No additional sequence scaler is fitted in V1.
- On-disk dtype: float32 NPY memory map, matching the existing builder.
- Device compute: AMP float16 with float32 losses and GradScaler. On V100 the
  phase-local scaler starts at `128` and does not grow within a phase; any
  overflow is a hard failure rather than a silently skipped optimizer update.

S2 SSL targets are future GMV sums at horizons 7, 14, and 30 days. For each
horizon it uses `buy_h = 1[gmv_h > 0]` and `gmv_z_h = log1p(gmv_h)`.

### ETT event-time input

- Maximum events: 180, ordered causally up to and including the anchor.
- Content: `[B, 180, 12]`, FP16 on disk, float32/autocast on device.
- Time features: `[B, 180, 12]`, FP16 on disk.
- Ranks: `[B, 180]`, int16 on disk and int64 on device.
- Padding mask: `[B, 180]`, bool.
- Empty-history flag: `[B]`, bool.
- No event after the anchor may enter any tensor.

## 5. Model classes

The initial implementation uses these existing class contracts. Architecture
changes require a new experiment ID.

### CatBoost

- `catboost.CatBoostClassifier` for React and Churn.
- `catboost.CatBoostRegressor` for Amount.
- Three independent models; no shared base and no SSL.
- Per model: 1500 iterations, depth 6, learning rate 0.04,
  `l2_leaf_reg=6`, seed derived from root seed, `task_type="GPU"`,
  `devices="0"`, `boosting_type="Plain"`, `grow_policy="SymmetricTree"`,
  `bootstrap_type="Bayesian"`, `bagging_temperature=1`, `random_strength=1`,
  `border_count=128`, `nan_mode="Min"`, and `allow_writing_files=False`.
- Losses: Logloss for React/Churn and RMSE for Amount.
- No class weights, sample weights, feature subsampling, early stopping, or use
  of M or V as an eval set.
- RUN A concatenates its 11 frames in chronological order without deduplication;
  RUN B independently concatenates its 14 frames the same way.
- A single `catboost.Pool` is created directly from each task-filtered Polars
  frame through one Pandas conversion. An intermediate NumPy feature copy is
  forbidden.
- React and Churn predictions are full-cohort raw formula logits. Amount is a
  full-cohort z prediction clipped to `[0, +inf)`.
- The three task models are trained sequentially and released immediately after
  holdout prediction.
- CatBoost version, CUDA runtime, GPU model, full resolved parameter dictionary,
  task row counts, seed, duration, best/last iteration, and prediction hashes are
  written to `model_training_report.json`.
- GPU CatBoost is not promised to be bitwise deterministic across driver or
  library versions. Reproducibility requires the recorded environment versions
  and artifact hashes; silent CPU fallback is forbidden.

### S1 Masked GRU

- `GRUBackbone`: two-layer GRU, input 15, hidden 128, dropout 0.2,
  followed by `TemporalAttention(128)`.
- SSL wrapper: `S1MaskedPretrainer`.
- SSL corruption: independently mask 20% of daily positions and zero the first
  12 behavioural channels at those positions.
- SSL head reconstructs all 15 channels; Smooth L1 is calculated only at masked
  positions.
- Transition model: `TransitionBase` with React, Churn, direct amount, and
  conditional amount heads.

### S2 Multi-Horizon GRU

- The same `GRUBackbone` architecture as S1, initialized independently.
- SSL wrapper: `S2MultiHorizonPretrainer`.
- SSL heads: buy logits and GMV-z predictions for horizons 7, 14, and 30.
- SSL loss:
  `0.55 * mean(BCE buy heads) + 0.45 * mean(SmoothL1 GMV heads)`.
- Transition model: a separate `TransitionBase` with the same four heads as S1.

### Event-Time Transformer

- `EventTimeTransformer`.
- Content projection 12 to 128, time projection 12 to 128, rank embedding 181
  by 128, LayerNorm.
- Two pre-norm Transformer encoder layers, four attention heads, feed-forward
  size 512, GELU, dropout 0.1.
- Final non-padding/last-token representation as defined by the existing class.
- React, Churn, direct amount, and conditional amount heads.
- No separate SSL stage in V1.

## 6. Exact optimizer budgets

The budgets below apply independently and identically in RUN A and RUN B.

| Model | Stage | Exact optimizer steps | Effective batch | Optimizer and LR |
|---|---|---:|---:|---|
| S1 | masked SSL | 750 | 512 | AdamW, 1e-3, wd 1e-4 |
| S1 | joint transition base | 2000 | 512 | AdamW, 5e-4, wd 1e-4 |
| S1 | each specialist H | 400 | 512 | AdamW, 1e-3, wd 1e-4 |
| S1 | each specialist F | 600 | 512 | AdamW, 1e-4, wd 1e-4 |
| S2 | multi-horizon SSL | 750 | 512 | AdamW, 1e-3, wd 1e-4 |
| S2 | joint transition base | 2000 | 512 | AdamW, 5e-4, wd 1e-4 |
| S2 | each specialist H | 400 | 512 | AdamW, 1e-3, wd 1e-4 |
| S2 | each specialist F | 600 | 512 | AdamW, 1e-4, wd 1e-4 |
| ETT | joint transition base | 3500 | 512 | AdamW, 3e-4, wd 1e-4 |
| ETT | each specialist H | 400 | 512 | AdamW, 1e-3, wd 1e-4 |
| ETT | each specialist F | 600 | 512 | AdamW, 1e-4, wd 1e-4 |

ETT effective batch 512 is implemented as micro-batch 128 with gradient
accumulation 4. An ETT optimizer step is counted only after `optimizer.step()`.
S1/S2 do not use gradient accumulation.

All neural stages use gradient clipping at norm 1.0. SSL and base stages use
linear warmup for the first 10% of steps followed by cosine decay to zero.
Specialist H and F phases use cosine decay without warmup. Scheduler steps are
optimizer steps, never micro-batches.

Per run, the neural budget is exactly:

```text
S1: 750 + 2000 + 3 * (400 + 600) = 5750
S2: 750 + 2000 + 3 * (400 + 600) = 5750
ETT:       3500 + 3 * (400 + 600) = 6500
Total per run: 18000 optimizer steps
Total RUN A + RUN B: 36000 optimizer steps
```

## 7. Joint base loss

S1, S2, and ETT use the same `transition_loss` contract:

- factorized prediction:
  `p_buy * max(conditional_z, 0)`;
- `p_buy = sigmoid(reactivation_logit)` for inactive users;
- `p_buy = sigmoid(-churn_logit)` for active users;
- MSE of factorized z versus `z_target`;
- weight 0.25 for direct-z MSE;
- weight 0.25 for conditional-z MSE on positive targets;
- weight 0.10 for React BCE on inactive users;
- weight 0.10 for Churn BCE on active users.

Changing these weights or definitions requires a new experiment ID.

## 8. Specialist scheme

Specialists are required in RUN A because meta must be fitted on the same kind of
predictions that RUN B supplies. Fitting meta on base heads and applying it to
fine-tuned specialist heads is forbidden.

Each neural base produces three independent specialists:

| Specialist | Training subset | Target | Output column suffix |
|---|---|---|---|
| React | `was_active == 0` | `will_buy` | `react_logit` |
| Churn | `was_active == 1` | `1 - will_buy` | `churn_logit` |
| Amount | `future_gmv_30d > 0` | `log1p(future_gmv_30d)` | `amount_z` |

Every specialist starts from the completed base encoder and a newly initialized
task head.

- Phase H: encoder completely frozen; train only the new head for 400 steps.
- Phase F for S1/S2: train the head, second GRU layer, and temporal attention;
  keep the first GRU layer frozen for 600 steps.
- Phase F for ETT: train the head, final Transformer layer, and final LayerNorm;
  keep projections and the first Transformer layer frozen for 600 steps.

Specialists are trained sequentially, not kept as three simultaneous GPU copies.
Immediately after a specialist is trained, predict the holdout, write its column
and checkpoint, release the model, and empty the CUDA cache.

The prediction schema is exactly:

```text
cb_react_logit, cb_churn_logit, cb_amount_z
s1_react_logit, s1_churn_logit, s1_amount_z
s2_react_logit, s2_churn_logit, s2_amount_z
ett_react_logit, ett_churn_logit, ett_amount_z
```

Every column must contain exactly 100000 finite values in cohort order.

## 9. RUN A meta optimizer

Meta is fitted once on the 100k predictions at M. There is no temporal OOF loop
and no first-level retraining inside meta optimization.

Feature order is fixed as CatBoost, S1, S2, ETT separately for React, Churn, and
Amount. Amount columns are standardized on M and the mean/scale are serialized.

For parameter vector
`[react_weights(4), churn_weights(4), amount_weights(4), amount_intercept]`:

```text
p_react = sigmoid(react_logits @ react_weights)
p_churn = sigmoid(churn_logits @ churn_weights)
p_buy = p_react                    when was_active == 0
p_buy = 1 - p_churn                when was_active == 1
conditional_z = max(0, standardized_amount @ amount_weights + intercept)
prediction_z = p_buy ** 1.1 * conditional_z
```

Optimizer contract:

- `scipy.optimize.minimize(method="SLSQP")`;
- objective: mean squared error between `prediction_z` and
  `log1p(future_gmv_30d)` on all M users;
- React weights: each in `[0, 1]`, sum exactly 1;
- Churn weights: each in `[0, 1]`, sum exactly 1;
- Amount weights: nonnegative, no sum constraint;
- Amount intercept: unbounded;
- alpha fixed at 1.1 and not optimized;
- one canonical start `[0.25 x4, 0.25 x4, 1.0 x4, 0.0]` plus eight
  seed-42 random starts;
- `maxiter=1000`, `ftol=1e-10`;
- accept only finite results with `success=True`; choose the lowest objective;
- serialize feature order, scaler, all starts, final parameters, objective,
  prediction-bank SHA256, code commit SHA, and config SHA256.

RUN B loads this package read-only. Re-fitting, calibration, threshold selection,
or choosing a different start using V labels is forbidden.

## 10. Execution order

```text
build shared stores once

RUN A:
  CatBoost specialists -> predictions on M
  S1 SSL -> base -> React/Churn/Amount specialists -> predictions on M
  S2 SSL -> base -> React/Churn/Amount specialists -> predictions on M
  ETT base -> React/Churn/Amount specialists -> predictions on M
  assemble immutable M prediction bank
  fit and freeze joint meta package
  destroy all RUN A model state

RUN B:
  initialize everything from scratch
  repeat the identical training recipe on the 14 RUN B anchors
  predict V with all first-level specialists
  assemble immutable V prediction bank
  apply the frozen RUN A meta package
  only then open V targets and calculate diagnostics
```

## 11. Required artifacts and diagnostics

The run-scoped output root is
`artifacts/ssl_temporal_stack_v1/post_ny_public_proxy/`. Existing Public,
record-recipe, PRE_NY, and `reference_pipeline_v1` artifacts are read-only and
must not be overwritten.

Required downloaded outputs:

```text
run_manifest.json
model_training_report.json
run_a_meta_prediction_bank.parquet
frozen_meta_package.json
run_b_validation_prediction_bank.parquet
validation_report.json
artifact_sha256.json
```

`model_training_report.json` records for every model/stage/task/phase:
requested and completed optimizer steps, effective examples, per-anchor batch
counts, elapsed seconds, final loss, peak GPU memory, AMP status, and seed.

`validation_report.json` contains overall RMSLE/MSE, 00/01/10/11 counts and
metrics, React/Churn/Amount diagnostics, target/prediction means, job ID, commit
SHA, config hash, prediction-bank hashes, and frozen-meta hash.

Feature frames, dense tensors, event memory maps, caches, and checkpoints remain
on the VM and are not DataSphere outputs.

## 12. Gates before a full job

1. Unit tests for exact-step completion, round-robin anchor coverage, specialist
   masks, causal tensors, and frozen-meta loading.
2. Local `py_compile` and YAML parsing in `myenv`.
3. A local 100-user dry run with tiny overridden steps; overrides are smoke-only
   and written to its manifest.
4. A separate DataSphere smoke job proving CUDA, CatBoost GPU, AMP finite losses,
   FP16 event memory maps, streaming progress, and output isolation.
5. A PRE-RUN commit containing the implementation, this specification, exact
   experiment config, and DataSphere manifest.
6. The full job is launched only from that PRE-RUN SHA through
   `scripts/datasphere_runner.py`.

Any difference from this document aborts the full launch or requires a new
versioned experiment specification.

## 13. Future roadmap — not part of this experiment

The following work is explicitly deferred. None of it may be silently added to
`SSL_TEMPORAL_STACK_V1`; every stage receives its own config, output root,
PRE-RUN commit, DataSphere job, RESULT commit, prediction-bank hashes, and
validation comparison.

1. **BTYD CatBoost features.** Create a separate
   `SSL_TEMPORAL_STACK_V1_BTYD` experiment. Add leakage-safe BTYD features only
   to CatBoost, freeze their fit scope and ordered feature contract, and compare
   against the completed SSL baseline before changing any neural model.
2. **First-level framework refactor.** After a completed immutable baseline,
   move CatBoost/S1/S2/ETT behind a registry/adapter API. Require prediction-bank
   parity on a fixed smoke fixture and unchanged training contracts before the
   refactor is allowed to become an experiment base.
3. **TCN candidate.** Add an independent TCN/dilated causal 1D CNN first-level
   model on the 180-day daily tensor. Its initial outputs are React and Churn
   only. It is trained in both RUN A and RUN B and contributes its own columns
   directly to joint meta; it must not consume predictions of existing models.
4. **Residual MLP candidate.** After evaluating TCN, add an independent residual
   MLP on tabular/aggregated features with React, Churn, and Amount outputs. It
   also contributes independent first-level columns and does not use CatBoost or
   stack predictions as inputs.

Required evaluation order:

```text
SSL baseline
-> SSL baseline + BTYD CatBoost features
-> parity-safe framework refactor
-> refactored baseline + TCN
-> refactored baseline + TCN + Residual MLP
```

Only one material change is evaluated at each transition. A later stage cannot
start until the preceding result and artifact hashes are frozen.
