# Public-LB calibration plan

This plan calibrates a future validation recipe, not the lost historical training
lineage of `ARTIFACT_RECORD_100K`.

Compare only like-for-like `VALIDATION_RECIPE_ANALOGUE` runs:

| Variant | Reported Public LB | Validation analogue |
| --- | ---: | --- |
| 100k Joint RMSLE | 1.6640779122 | REFERENCE_PIPELINE_V1 + joint meta fitting |
| 100k Separate meta | 1.6649359945 | same V1 first-level predictions + separate React/Churn/Amount fitting |
| 250k exact | 1.6657399037 | excluded until its full lineage is confirmed |

For Joint versus Separate, hold cohort, anchors, first-level models, preprocessing,
seed and alpha fixed.  Record validation RMSLE delta, reported Public delta,
direction/ranking agreement, state deltas, prediction distribution, and confounders.
The key question is whether validation ranks Joint above Separate, as reported on
Public LB.  With only two points, do not calculate or interpret a correlation;
with three, do not make a strong correlation claim either.

If ranking differs, treat the protocol as uncalibrated and inspect temporal split,
meta/final separation, implementation differences and seasonal regime before any
submission decision.  Public LB is never an optimization target.
