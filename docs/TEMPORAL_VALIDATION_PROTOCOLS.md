# Temporal validation protocols

## Two different objects

`ARTIFACT_RECORD_100K` is the existing Public-LB record.  Its CSV can be rebuilt
from the retained bank and joint JSON with a maximum numerical delta of about
`2.3e-12`; its original training lineage is partially lost.

`REFERENCE_PIPELINE_V1` is a future, fully versioned implementation.  It must not
be described as an exact reconstruction of the record training run until its
S1/S2/ETT source, preprocessing, checkpoints, cohort provenance, and configs have
been resolved.  This document designs validation for the latter.

## Fixed baseline contract

The first reference baseline must use the immutable `selected_users_100k.parquet`
(SHA-256 `d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2`),
without resampling; 90-day history; 30-day future target; and:

```text
activity_definition = record_legacy_90d_activity
was_active = int(GMV in the available 90-day snapshot history > 0)
z_target = log1p(future_gmv_30d)
alpha = 1.1
model_order = CatBoost, S1, S2, ETT
React/Churn inputs = raw logits
Amount inputs = conditional z-space predictions
```

`was_active_30d` is a separate hypothesis.  It changes specialist subsets,
transition labels, routing and meta rows, so it requires two complete matched runs;
it cannot be tested by switching a bank column.

## Label and anchor availability

The canonical manifest has 23 anchors from `2025-03-31` to `2026-01-14` and raw
data labels through `2026-02-13`.  For anchor `A`, target is `[A+1, A+30]`.

An admissible pair `(M, V)` satisfies:

```text
M < V
target_end(M) <= V
target_end(V) <= 2026-02-13
```

This yields **208** admissible pairs.  The exact compact enumeration is the
canonical anchor sequence below: for a validation anchor at index `j`, every earlier
meta anchor at index `i` with `anchor[i] + 30 days <= anchor[j]` is admissible.
For this sequence it is normally `i <= j - 3`; irregular December spacing is
checked by date arithmetic, not by index alone.  For every pair:

- RUN A anchors are all `A` with `target_end(A) <= M`;
- RUN B anchors are all `A` with `target_end(A) <= V`;
- neither `M` nor `V` is a training row of its own run;
- both target windows are available in existing raw labels and snapshots.

This is the full pair matrix without inventing dates; its selected and operational
subsets are stored in `configs/validation/protocol_candidates.yaml`.

## Required validation mode

### `FULL_TWO_RUN_VALIDATION`

1. **RUN A / meta**: train only on anchors whose targets end no later than `M`;
   create first-level predictions for `M`; fit and freeze the meta package using
   only target `[M+1, M+30]`.
2. **RUN B / final**: train fresh base models and fresh specialists on anchors whose
   targets end no later than `V`; infer on `V`; apply the frozen package from RUN A;
   evaluate only against `[V+1, V+30]`.

Neural checkpoints from RUN A must not be reused in RUN B.  This mirrors the
production sequence `META RUN -> fixed package -> FINAL RUN -> inference`.

`CHEAP_CARRY_FORWARD_VALIDATION` may retain RUN A models for inference on `V`, but
is diagnostic only: it has lower cost but is not equivalent to production and
cannot select a submission candidate.

## Selected profiles

| Profile | M | M target | RUN A anchors | V | V target | RUN B anchors | Purpose |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| `PRE_NY_PRIMARY` | 2025-10-13 | 2025-10-14..2025-11-12 | 12, last 2025-09-01 | 2025-11-24 | 2025-11-25..2025-12-24 | 15, last 2025-10-13 | historical baseline |
| `NY_CROSSING_STRESS` | 2025-10-27 | 2025-10-28..2025-11-26 | 13, last 2025-09-15 | 2025-12-08 | 2025-12-09..2026-01-07 | 16, last 2025-10-27 | seasonal stress |
| `POST_NY_PUBLIC_PROXY` | 2025-12-15 | 2025-12-16..2026-01-14 | 17, last 2025-11-10 | 2026-01-14 | 2026-01-15..2026-02-13 | 20, last 2025-12-15 | sealed confirmation |

All selected pairs have no target overlap.  Their M-to-V distances are respectively
42, 42, and 30 days.  V is respectively 81, 67, and 30 days before the final test
anchor `2026-02-13`.

`PRE_NY_PRIMARY` remains a historical baseline because both targets end before
New Year. `POST_NY_PUBLIC_PROXY` is the active reference profile because it is
closest to the final operational timing; it is treated as sealed rather than an
everyday tuning score. Its RMSLE must not be averaged with other seasonal profiles
without an explicit business reason.

Decision structure: a candidate improves primary development; does not show a
catastrophic seasonal regression; then receives one sealed post-NY confirmation.
No numerical acceptance thresholds are set before observing reference results.

## Metrics and anti-overfitting rules

Final decision metrics are MSE in `log1p` space and RMSLE.  Save prediction mean,
median, P90/P95/P99/max, log bias, shares `<1`, `>100`, `>1000` RUB; state
00/01/10/11 row count/share/MSE/RMSLE/bias/loss share/target and prediction means
and medians; React/Churn AUC+LogLoss; positive-only Amount MSE; and distributions
of `p_buy` and `conditional_z`.

Do not tune alpha, meta weights, post-processing or final checkpoints on V labels;
do not call user-level CV temporal validation; do not use Public LB in an optimizer;
and do not repeatedly inspect the sealed profile.  Any checkpoint selection must
be an inner protocol ending before M.
