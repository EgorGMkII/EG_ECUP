# RESULT — direct CatBoost screen (25k)

## Decision

Do not promote `catboost_direct` into the selected hurdle protocol.  The
standalone direct model is a useful control, but the leakage-safe late blend
is worse than the immutable hurdle baseline on the held-out validation window.

## Immutable run identity

- DataSphere job: `bt1ulpo9ipdvfps1q4pv` (`SUCCESS`)
- PRE-RUN commit: `afb4f23b6d71d13780774c7d1a9d5f31db06b576`
- Config: `post_ny_screen_direct_catboost_v1`
- Config SHA-256: `06ae7f8653614c08c859e87afb260ed9a23759d430ee20a8ed150ef817257450`
- Downloaded candidate artifact SHA-256: `c3c6822623aa62410c4602af024cbc95b816a3d227dca33eb1256515238b9b5e`
- Profile/cohort: `POST_NY_PUBLIC_PROXY`, deterministic 25k screen cohort
- RUN A/M: `2025-12-15`; RUN B/V: `2026-01-14`
- V target: `2026-01-15 .. 2026-02-13`

## Direct model

- `CatBoostRegressor`, GPU, 300 trees, depth 8, LR 0.05, L2 5.
- Target: `log1p(future_gmv_30d)`.
- 75 sparse causal features over 7/14/30/60/90/180/365 days.
- For each holdout `T`, training snapshot is `T-30d`; its labelled target
  ends exactly at `T`.  The recorded gap is zero in both runs.

## Scores

| Evaluation | RMSLE | MSE (log-space) |
| --- | ---: | ---: |
| Direct standalone, RUN A/M | 1.7609836732 | 3.1010634972 |
| Direct standalone, RUN B/V | 1.7031833035 | 2.9008333652 |
| Immutable hurdle baseline, RUN B/V | 1.6875360759 | 2.8477780074 |
| Hurdle + frozen RUN-A late blend, RUN B/V | 1.6904344605 | 2.8575686653 |

The M-fitted late blend assigned `0.93532585` to direct CatBoost and
`0.06467415` to hurdle, then failed to transfer to V.

## Paired V comparison

- Candidate minus baseline MSE: `+0.0097906579`
- Approximate RMSLE delta: `+0.0028983846`
- 5,000-repeat paired-bootstrap 95% CI for MSE delta:
  `[-0.0018587179, +0.0218543559]`
- Probability candidate is better: `0.0494`

This is an unpromoted screen result.  It must not replace the baseline frozen
meta package or alter historical/Public artifacts.
