# RESULT — direct CatBoost full 100k confirmation

## Decision

Promote the direct CatBoost late-blend channel for a separate 250k final
submission. This does **not** replace the no-direct six-model submission;
both variants remain separately pinned and reproducible.

## Immutable identity

- DataSphere job: `bt1qi0pvel43dvb75ijm` (`SUCCESS`)
- PRE-RUN commit: `833a02e325f254e2cec077daa3ef8feb992b2e9c`
- Direct config SHA-256: `67f9f7f8dc5977b4868f3715922ccbe3d1654e129682d9a03da9f7b2add77ba3`
- Direct artifact SHA-256: `71a5b807f4ccbb9e3f5f0bdc452646abd3df2b43989945d9d833c76c80e1daed`
- Baseline: `post_ny_ssl_parity_tcn_mlp_btyd_selected_100k_v1`
- Profile/cohort: `POST_NY_PUBLIC_PROXY`, paired selected 100k cohort
- V target: `2026-01-15 .. 2026-02-13`

## Direct recipe

- GPU `CatBoostRegressor`, 300 trees, depth 8, LR 0.05, L2 5.
- Direct target: `log1p(future_gmv_30d)`.
- 75 sparse causal features, 7/14/30/60/90/180/365-day windows.
- For each holdout `T`, training snapshot is `T-30d` and its label window
  ends exactly at `T`; the recorded gap is zero in RUN A and RUN B.

## Results on the paired 100k V bank

| Variant | RMSLE | MSE (log-space) |
| --- | ---: | ---: |
| Six-model BTYD+TCN+MLP hurdle baseline | 1.6793943963 | 2.8203655384 |
| Direct standalone | 1.6913583454 | 2.8606930524 |
| Hurdle + frozen RUN-A direct late blend | 1.6776567210 | 2.8145320735 |

The RUN-A meta optimizer selected late-blend weights: direct `0.92506504`,
hurdle `0.07493496`.

## Paired bootstrap

- Candidate minus baseline MSE: `-0.0058334649`
- RMSLE delta: `-0.0017376753`
- 5,000-repeat paired-bootstrap 95% CI for MSE delta:
  `[-0.0109299751, -0.0007321962]`
- Probability candidate is better: `0.9888`

This is the full-cohort promotion gate. The two final configurations must pin
the no-direct full meta package and this incremental direct package
respectively; neither historical artifact is overwritten.
