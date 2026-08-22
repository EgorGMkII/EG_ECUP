# Neural implementation forensics

## Scope and terminology

`ARTIFACT_RECORD_100K` is the existing record artifact.  `REFERENCE_PIPELINE_V1`
is a reproducible future analogue, not a claim that its new training will recreate
the exact historical bank.  Evidence below therefore distinguishes source facts
from a provenance link to that bank.

## S1/S2 finding

The project contains a confirmed historical distinction:

- **S1** pretrains a common GRU+temporal-attention encoder with masked behavioural
  reconstruction.  It masks roughly 20% active/day spans and reconstructs all 15
  daily channels.
- **S2** pretrains the same encoder type on dense multi-horizon future objectives:
  buy, log-GMV and purchase days at 7/14/30 days.

Both reset `torch.manual_seed(42)`, use a two-layer GRU-180 (`input=15`,
`hidden=128`, dropout `0.2`), pretrain for four epochs with AdamW `1e-3`, transfer
the encoder GRU+attention strictly into `MultiTaskTransitionGRUModel`, then
fine-tune for ten epochs.  The downstream loss and 11 purged anchors are common.
The difference is the pretraining objective, not GRU class, input tensor, anchors,
seed, or downstream specialist head.

| Variant | Source path | Encoder | Input representation | Masking | Anchors | Steps | Seed | Evidence |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| S1 masked behaviour | `scripts/run_ssl_pretraining_experiments.py` | `GRUEncoder` + `TemporalAttention` | dense daily `[B,180,15]` | 20% active/span corruption; reconstruction loss only on masked positions | 11, 2025-07-21..2025-12-08 | 4 pretrain epochs + 10 downstream | 42 | `CONFIRMED_BY_CODE`; metrics artifact confirms named run |
| S2 dense multi-horizon | same | same | dense daily `[B,180,15]` | no reconstruction mask; predictive 7/14/30 heads | same 11 | 4 pretrain epochs + 10 downstream | 42 | `CONFIRMED_BY_CODE`; metrics artifact confirms named run |
| Record builder S1 | `scripts/build_joint_rmsle_submission.py` | `GRUEncoder` | event-style `[B,180,12]` content plus unused time/rank/mask args | no GRU masking behaviour | eight final anchors | 3000 | 42 process seed | `CONFIRMED_BY_CODE`, but bank linkage `INFERRED` |
| Record builder S2 | same | same class | identical to builder S1 | identical | identical eight anchors | 3000 | sequential RNG after S1 | `CONFIRMED_BY_CODE`, but bank linkage `INFERRED` |
| RUN 1 ETT legacy | `scripts/run1_train_meta_weights.py` | ETT FF256 | event tensors | learned empty token | 17 meta-run anchors | source-defined | 42 | `CONFIRMED_BY_CODE`; legacy only |

The immutable record bank has separate S1/S2 columns, but that fact alone does not
prove their training implementation.  There is no final record checkpoint or run
log binding either variant to the bank: that link remains `UNKNOWN`.

## Recommendation for REFERENCE_PIPELINE_V1

Use two explicitly distinct models:

```text
s1_masked_behavior_ssl_gru180_v1
s2_dense_multihorizon_ssl_gru180_v1
```

They are evidence-backed variants with common daily tensor preprocessing and fixed
seed/objective contracts.  They are not claimed to be exact record-bank S1/S2.
The alternative record-builder labels `Masked`/`Dense` must not be used as semantic
evidence because that code creates two identical `GRUEncoder` classes; only their
RNG trajectory differs.

## Canonical ETT for REFERENCE_PIPELINE_V1

Use `ett_record_builder_last_token_ff512_v1` from
`scripts/build_joint_rmsle_submission.py::EventTimeTransformer`:

```text
d_model=128; n_heads=4; n_layers=2; dim_feedforward=512; max_events=180
content projection + time projection + rank embedding + LayerNorm
last-token pooling
all-masked row: unmask token 0 before Transformer, then zero its output embedding
heads: ReLU direct/conditional z; raw React/Churn logits
```

Its extractor consumes the most recent 180 events from a one-year history and
creates 12 content plus 12 time features, including tau=30 decay and day-of-year
sin/cos.  This is `CONFIRMED_BY_CODE` as the selected source implementation;
its direct causal relation to the historical Public-LB improvement is not proven.

The FF256 learned-empty-token, last+mean+max pooling implementation in
`scripts/run1_train_meta_weights.py` remains `ett_run1_ff256_pooling_legacy_v1`.
It is retained as legacy and must not become canonical by accident.
