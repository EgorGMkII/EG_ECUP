# Record 100k reproducibility specification

## Scope and status

This document separates two different claims.

- **Artifact-level reproduction** means deterministic assembly of the record CSV
  from its immutable raw-specialist prediction bank and fixed meta JSON.
- **Training-level reproduction** means creating that same bank again by training
  the first-level models from raw events.

Only the first claim is verified in this repository state.  No model training,
feature generation, or inference was performed for this specification.

## Immutable record inputs

| Item | Path | SHA-256 |
| --- | --- | --- |
| Record submission | `submission_specialized_hurdle_joint_rmsle.csv` | `3300512c94579fc6692efb3a6d51a160f0ae5f2375c1476c3aaa54ff775aedcd` |
| Raw prediction bank | `test_specialists_raw_predictions_250k.parquet` | `ddb0e882d80f002752f95d10388df40f09a7bebb3d3e61f92153a1a99fdab0d0` |
| Joint meta JSON | `artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json` | `e9077605f9b438311c46fa7151a099b617ff457eb5f87d972f465502c873961b` |
| Diagnostics | `submission_specialized_hurdle_joint_rmsle_diagnostics.parquet` | `ce53ad3ca39731c81842ac5820903d80736100f933711632d05cf5ff65e169fa` |
| Submission template | `sample_submit.csv` | `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1` |

The reported external Public LB value is `1.66407791221865`.  It is a reported
external result, not independently verified from a leaderboard in this step.

The record CSV has 250,000 rows and exact schema `user_id,predict`.  Its prediction
distribution is: mean `43.21015669699452`, median `7.649151609298004`, P90
`110.80734252436949`, P99 `540.1434726093125`, max `3004.7916282966457`.

## Prediction-bank contract

The bank has 250,000 unique `Int64` users and these columns in this exact order:

```text
user_id: Int64
was_active: Int64
cb_react_logit: Float64     s1_react_logit: Float32
s2_react_logit: Float32    ett_react_logit: Float32
cb_churn_logit: Float64     s1_churn_logit: Float32
s2_churn_logit: Float32    ett_churn_logit: Float32
cb_amount_z: Float64        s1_amount_z: Float32
s2_amount_z: Float32       ett_amount_z: Float32
```

There are no null or NaN values in the bank.  All twelve prediction columns have
non-zero variance.  The fixed model order is:

```text
CatBoost, S1_GRU, S2_GRU, ETT
```

`*_react_logit` and `*_churn_logit` are raw logits by contract: the generating
code uses CatBoost `RawFormulaVal` and neural head outputs before sigmoid.  The
assembler rejects probability-suffixed columns rather than inferring semantics
from a numeric range.  `*_amount_z` are conditional Amount values in
`z = log1p(GMV)` space, not rubles and not standardized values.  No Amount scaler
is part of the 100k record JSON.

## Record semantics

### Record semantics: `record_legacy_90d_activity`

For the record recipe, the executable scripts compute:

```python
was_active_90d = int(lifetime_gmv > 0)
```

In snapshots, `lifetime_gmv` is calculated from the available 90-day snapshot
history.  Despite its name, it is not complete lifetime GMV.  The bank contains
61,665 inactive and 188,335 active users under this legacy definition.

### Alternative semantics: `was_active_30d`

```python
was_active_30d = int(gmv_sum_30d > 0)
```

This is a future experimental alternative only.  It is not part of the record and
must not silently replace `record_legacy_90d_activity` in a baseline run.

Snapshots use a 90-day history ending at anchor `A`; their supervised target is
GMV in `[A + 1 day, A + 30 days]`.  The test anchor is reported as `2026-02-13`,
with target window `2026-02-14` through `2026-03-15`.

## Canonical artifact-level meta formula

The exact weights are read from the immutable joint JSON:

```text
React:  [0.02571986359935257, 0.3658967641100093, 0.0, 0.6083833722906381]
Churn:  [0.2770564003031295, 0.22700033415688514, 0.06279933646526792, 0.43314392907471744]
Amount: [0.12217097079693888, 4.146865521979716e-19, 0.35635864495834946, 0.5454727117581657]
Amount intercept: 0.02079580550858867
ALPHA: 1.1
```

The assembler casts all prediction matrices and weights to `float64`, reconstructs
template order by an exact `user_id` join, then applies:

```python
p_react = expit(X_react @ w_react)
p_churn = expit(X_churn @ w_churn)
p_buy = np.where(was_active == 0, p_react, 1.0 - p_churn)
conditional_z = np.maximum(0.0, X_amount @ w_amount + amount_intercept)
z_prediction = np.clip(np.power(p_buy, 1.1) * conditional_z, 0.0, None)
predict = np.expm1(z_prediction)
```

It validates the JSON version/model order, fixed `ALPHA=1.1`, required columns,
unique users, complete user-set equality, NaN/Inf, and non-zero bank variance.
It never trains, builds features, or writes to old artifact paths.

## Verified artifact-level reproduction

The assembler is `scripts/rebuild_record_submission_from_bank.py`.  Its output
is written only to `artifacts/record_100k_reproduction/`.

| Check | Result |
| --- | --- |
| Row count and schema | exact match |
| `user_id` membership and order | exact match |
| Mean / median / P90 / P99 / max | exact match at displayed precision |
| Max absolute difference | `2.2737367544323206e-12` |
| Mean absolute difference | `4.816218479586354e-15` |
| Values differing at binary float comparison | 77,908 |
| Record CSV SHA-256 | `330051…aedcd` |
| Rebuilt CSV SHA-256 | `e8624f…93e6f` |
| Status | `NUMERICALLY_EQUIVALENT_REPRODUCTION` |

The difference is consistent with float representation and CSV formatting/order of
floating-point operations.  It is not evidence for a changed formula: complete
user alignment and all aggregate distribution checks match.  Therefore this is
not `BYTE_EXACT_REPRODUCTION`.

The generated `artifact_manifest.json` records input/output/script hashes; the
generated `verification.json` records the complete numerical comparison.

### Files sufficient for artifact-level reproduction

1. `test_specialists_raw_predictions_250k.parquet`
2. `artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json`
3. `sample_submit.csv`
4. `scripts/rebuild_record_submission_from_bank.py`

The original record CSV is needed only to verify the result, not to rebuild it.

## Training lineage of the record bank

Status vocabulary:

- `CONFIRMED_BY_CODE`: current source explicitly implements the behaviour.
- `CONFIRMED_BY_ARTIFACT`: bank/schema/config proves the stored contract.
- `CONFIRMED_BY_LOG`: a run log ties the implementation to the artifact.
- `REPORT_ONLY`: stated in reports without a binding code/artifact proof.
- `INFERRED`: supported by code/artifact names or matching structure, but not
  tied to this immutable bank by an execution log/checkpoint.
- `UNKNOWN`: required provenance is unavailable.

| Component | Generating implementation | Preprocessing / schedule | Exact lineage status |
| --- | --- | --- | --- |
| `was_active` | `lifetime_gmv > 0` in RUN 1, RUN 2 and builder | 90-day snapshot GMV; stored as bank `Int64` | `CONFIRMED_BY_CODE` + `CONFIRMED_BY_ARTIFACT` |
| CB React | Current builder/RUN scripts emit `RawFormulaVal` | pooled 100k snapshots; 23 final anchors; 1500 iterations | `INFERRED` |
| CB Churn | Current builder/RUN scripts emit `RawFormulaVal` | same; target is `1 - will_buy` on active users | `INFERRED` |
| CB Amount | Current builder/RUN scripts emit `log1p` regressor output | positive target rows; clamped at zero in builder | `INFERRED` |
| S1 React/Churn/Amount | `GRUEncoder` in current builder | 12 content features, 8 neural anchors, 3000 steps | `INFERRED` |
| S2 React/Churn/Amount | same `GRUEncoder` class in current builder | same input construction and 3000 steps | `INFERRED` |
| ETT React/Churn/Amount | `EventTimeTransformer` in current builder | 180 tokens, 12+12 features, decay tau 30, 4500 steps | `INFERRED` |
| Joint meta level | bank schema + joint JSON + formula | float64 assembly, fixed order and alpha | `CONFIRMED_BY_CODE` + `CONFIRMED_BY_ARTIFACT` |

No `CONFIRMED_BY_LOG` linkage exists from the immutable bank to a source revision
or a final neural checkpoint.  `artifacts/specialized_hurdle/logs/run_status.json`
still records initialization, and `job_manifest.csv` marks jobs pending.

### S1/S2/ETT uncertainty

The current record builder directly creates both S1 and S2 with the same
`GRUEncoder` class and the same extracted event-time buffers.  It does not expose
a separate masking implementation for S1 versus dense implementation for S2.
Thus distinct S1/S2 implementations for the run that made the immutable bank are
`UNKNOWN`; their difference is `REPORT_ONLY`.

There are two incompatible ETT implementations in source:

| Source | FF width | Pooling | Empty sequence handling |
| --- | ---: | --- | --- |
| `scripts/run1_train_meta_weights.py` | 256 | last + mean + max MLP | learned empty-history token |
| `scripts/build_joint_rmsle_submission.py` | 512 | last token | unmask token 0, then zero embedding |

Both use 2 layers, 4 heads, `d_model=128`, max 180 events, and tau 30 temporal
decay.  The bank itself does not encode architecture, pooling, or source revision,
so the exact ETT version is `UNKNOWN`.  Reports claiming FF width 512 are
`REPORT_ONLY` until a run log or checkpoint binds them to the bank.

The intended final neural anchors in current RUN 2/builder code are:

```text
2025-03-31, 2025-04-28, 2025-05-26, 2025-06-23,
2025-07-21, 2025-08-18, 2025-09-15, 2026-01-14
```

They are `CONFIRMED_BY_CODE` for the current script and `INFERRED` for the bank.
No final record S1/S2/ETT checkpoints exist.  The bank does not carry code hash,
training seed, sequence manifest, or checkpoint lineage.

## Training-level reproducibility assessment

| Component | Exact implementation known | Config known | Training users known | Anchors known | Checkpoint exists | Preprocessing known | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CB React | no | yes | inferred | inferred | no final checkpoint | partly | `INFERRED` |
| CB Churn | no | yes | inferred | inferred | no final checkpoint | partly | `INFERRED` |
| CB Amount | no | yes | inferred | inferred | no final checkpoint | partly | `INFERRED` |
| S1 React | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| S1 Churn | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| S1 Amount | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| S2 React | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| S2 Churn | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| S2 Amount | no | partly | inferred | inferred | no | no | `UNKNOWN` |
| ETT React | no | conflicting | inferred | inferred | no | conflicting | `UNKNOWN` |
| ETT Churn | no | conflicting | inferred | inferred | no | conflicting | `UNKNOWN` |
| ETT Amount | no | conflicting | inferred | inferred | no | conflicting | `UNKNOWN` |
| Joint meta-level | yes | yes | n/a | meta OOF known | JSON exists | yes | `CONFIRMED_BY_ARTIFACT` |

Overall status: **`TRAINING_LEVEL_NOT_REPRODUCIBLE`**.  The source is enough to
attempt a new training run, but not enough to establish that it will recreate the
record bank or the record submission.

## Baseline contract for future validation

The following must not change in a future validation run unless the run is declared
a new experiment rather than a record-baseline reproduction:

- selected 100k cohort and its SHA-256 `d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2`;
- `record_legacy_90d_activity`;
- 90-day history and 30-day target window;
- CatBoost feature schema/hash `93cd750609e0`;
- real S1/S2/ETT implementation and sequence preprocessing;
- specialist definitions, neural anchors and schedules;
- model order; logits rather than probabilities; Amount `z` space;
- joint optimization constraints, fixed `ALPHA=1.1`, clipping, and final formula.

Already confirmed: cohort file hash, legacy activity semantics, bank schema/order,
meta JSON/order/formula, alpha, clipping, and artifact-level result.  Must be
resolved before the first honest temporal validation run: exact source revision
that produced the bank; final neural checkpoints; definitive S1/S2 distinction;
definitive ETT architecture; feature-schema runtime assertion; and a cohort
selection provenance manifest.
