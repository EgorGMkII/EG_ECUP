# Reference framework V1 — hyperparameter screening protocol

## Purpose and guardrails

This protocol begins only after the current six-model 100k validation result
has been recorded. It selects at most two candidate protocols on
`screen_cohort_25k`, then confirms them with a completely fresh `RUN A -> RUN
B` on `full_cohort_100k`.

Every candidate in one series uses identical cohort membership, anchors, raw
data manifest, root seed, meta formula and `alpha = 1.1`. No run may use test
labels, Public LB, transition rebalancing, row downsampling, global state
weights, automatic early stopping or automatic best-checkpoint selection.

Do not tune history length, cohort size, transition-stratum distribution,
hidden/embedding dimensions, layer/head counts, feature channels, Amount
formulation, `alpha`, meta constraints, state-specific thresholds/calibration
or new features in this protocol. Batch sizes remain fixed in initial series.

## Execution order

Use sequential one-factor screening, never a Cartesian grid:

1. Fix a reference configuration and immutable screen cohort.
2. Change exactly one parameter or one related recipe group.
3. Run the entire fresh `RUN A -> RUN B` candidate.
4. Compare V RMSLE with the current series reference, including transitions.
5. Record a series summary and promote only an unambiguous leader.
6. Only after the final 25k series, confirm one or two candidates fresh on
   100k; select one final protocol by V RMSLE.

Small 25k deltas are hypotheses, not improvements, until 100k confirmation.

## Candidate series

### 1. Training duration and selected checkpoints

Highest priority. A checkpoint is a separate immutable config and must follow:

```text
base checkpoint -> specialists from that base -> RUN A meta -> fresh RUN B -> V
```

Do not mix specialists derived from different base checkpoints.

| Component | Candidate checkpoints |
|---|---|
| S1/S2 base | epochs 6, 8, 10, 12 |
| ETT base | optimizer steps 3000, 3750, 4500, 5250, 6000 |
| Specialists | final H; F at 1000, 1750, 2500, 3250 |

The current framework uses explicit optimizer-step recipes rather than epochs.
Before this series, add an explicit epoch-to-step mapping to the resolved
config and record both values; do not approximate epochs silently.

### 2. Base learning rate and schedule

Run separate series:

| Component | Values |
|---|---|
| S1/S2 base peak LR | `5e-4`, `1e-3` reference, `1.5e-3` |
| ETT base peak LR | `2e-4`, `3e-4` reference, `4e-4` |
| ETT warmup | `250`, `500` reference, `750` steps |

### 3. Specialist LR and duration

Keep H/F paired rather than forming a grid:

| H/F LR pair | H/F step budget |
|---|---|
| `(5e-4, 5e-5)` | short: `1000/2000` |
| `(1e-3, 1e-4)` reference | reference: `1500/2500` |
| `(2e-3, 2e-4)` | long: `2000/3000` |

Each pair and each duration is its own config. Do not adjust them from RUN B
results during a job.

### 4. Regularization

After duration/LR series:

| Parameter | Values |
|---|---|
| S1/S2 dropout | `0.10`, `0.20` reference, `0.30` |
| ETT dropout | `0.05`, `0.10` reference, `0.20` |
| Weight decay | `3e-5`, `1e-4` reference, `3e-4` |
| Gradient clipping, last | `0.5`, `1.0` reference, `2.0` |

### 5. S1/S2 SSL

After base schedule only. First test SSL budgets with reference masking:

| Model | SSL epochs |
|---|---|
| S1 | `0`, `2`, `4` reference, `8` |
| S2 | `0`, `2`, `4` reference, `8` |

If and only if S1 SSL is positive versus zero, separately test S1 mask
probability `0.10`, `0.20` reference, `0.30`.

### 6. Shared multitask loss profiles

Apply each profile equally to S1, S2 and ETT; do not create state-specific
losses or new loss terms.

| Profile | factorized | direct | conditional | React BCE | Churn BCE |
|---|---:|---:|---:|---:|---:|
| reference | 1.00 | 0.25 | 0.25 | 0.10 | 0.10 |
| classification_focus | 1.00 | 0.25 | 0.25 | 0.20 | 0.20 |
| amount_focus | 1.00 | 0.25 | 0.50 | 0.10 | 0.10 |

### 7. Limited CatBoost recipes

No broad CatBoost search. Preserve GPU, feature manifest, causal preprocessing,
seeds and no early stopping.

| ID | iterations | learning rate | depth | l2 leaf reg |
|---|---:|---:|---:|---:|
| CB_A reference | 1500 | 0.040 | 6 | 6.0 |
| CB_B longer_soft | 2000 | 0.030 | 6 | 6.0 |
| CB_C shallower | 1500 | 0.040 | 5 | 6.0 |
| CB_D stronger_regularization | 1500 | 0.040 | 6 | 10.0 |

## Required artifacts and monitoring

Each resolved config has immutable `config_id`, canonical JSON/YAML and SHA256.
Write append-only structured metrics to:

```text
artifacts/reference_v1/runs/<config_id>/<run>/<model>/<task>/metrics.jsonl
```

Every event must include timestamp, run, cohort, config ID/hash, architecture,
stage, task, epoch when applicable, optimizer step, LR, total and component
losses, gradient norm, allocated GPU memory and elapsed seconds.

At fixed checkpoints record train and monitor metrics. The monitor is M in RUN
A and V in RUN B, and is diagnostic only: it may never select a checkpoint,
stop training or alter a schedule inside the job. Report rows and separate
00/01/10/11 metrics. Base reports additionally include factorized/direct/
conditional Amount error, React and Churn BCE/AUC/LogLoss. Specialist reports
include task loss, monitor loss, prediction mean and standard deviation.

After meta is available, record end-to-end MSE/RMSLE in z-space, log bias,
loss shares, prediction distribution and transition breakdown. RUN B V RMSLE
is the candidate's sole selection metric.

Automatically create:

- `training_curves.png` — train/monitor loss and RMSLE over steps;
- `specialist_curves.png` — React/Churn/Amount task curves;
- `overfitting_summary.json` — fixed-checkpoint train-monitor gaps and an
  observed worsening flag; it never changes the run automatically;
- `run_summary.json` — fully resolved recipe, hashes, M/V metrics, transitions
  and status.

## Series result table and promotion

After every series, append a table with:

```text
config_id, changed_parameter, value, cohort,
RUN_A_M_RMSLE, RUN_B_V_RMSLE, V_delta_vs_reference,
RMSLE_00, RMSLE_01, RMSLE_10, RMSLE_11, elapsed_seconds, status
```

Advance to the next group only after this summary exists. Do not transfer a
winning value between architectures without a dedicated run. At the end of
screening, run one or two complete protocols fresh on 100k, choose one by V
RMSLE, and only then train that pinned protocol for the 250k submission.
