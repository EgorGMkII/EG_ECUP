# Direct temporal CV v1 — implementation handoff

## Objective

Fill `src/direct_temporal_cv_v1/` without modifying historical Public builders,
`ssl_temporal_stack_v1`, or `reference_framework_v1`. The first gate is a
literal four-fold 250k reproduction of the supplied direct CatBoost protocol.

## Protocol

`FOUR_FOLD_250K_V1` is the only active protocol. Each fold trains one new model
from one snapshot at `T - 30 days`, with target `(T-30, T]`, then predicts from
the independently built feature snapshot at `T` for target `(T, T+30]`.
All four folds use all template users in template order. No pooled anchors,
RUN A/RUN B, specialists, row weighting, or random split are allowed.

## Fill order

1. `datasets.build_target_z`: causal inclusive target extractor, template-aligned.
2. `SparseAggregateFeatureProvider`: seven windows, sparse aggregation only,
   explicit feature manifest and leakage audit.
3. `DirectCatBoostAdapter`: CPU CatBoost with the exact YAML recipe.
4. `pipeline.run_cross_validation`: one fold at a time; save bank, metrics,
   resolved config, and SHA-256 manifests; release fold data afterwards.
5. Validate the four individual scores and mean against the reported `1.717960`.
   Stop here if feature/contract parity is not credible.
6. `BTYDFeatureProvider`: only `p_buy_30d`, expected purchases and p_alive;
   never a target-derived column.
7. Direct ETT then direct TCN: fresh model per fold, one scalar z head, no SSL,
   no H/F specialists and no transition targets.
8. Only after independent banks exist, implement `blending.py`: fit weights on
   F1--F3 only and report F4 separately as the untouched gate.

## Required artefacts

Each run writes a non-overwriting root under
`artifacts/direct_temporal_cv_v1/experiments/<experiment_id>/` with resolved
config, protocol/feature manifests, each fold's predictions and metrics, a CV
summary, model reports, and hashes. Do not export temporary feature caches,
daily tensors, event memmaps, checkpoints, or raw data.

## Non-negotiable integrity checks

- Features use only `event_date <= anchor`.
- Target columns cannot enter model features.
- All 250k template IDs occur exactly once and retain template order.
- `train_target_end == inference_anchor` and validation starts the next day.
- F4 labels/predictions never participate in blend fitting or candidate choice.
- Models, optimizers, schedulers and scalers are recreated for every fold.
- A missing user has zero target, never a dropped row.
