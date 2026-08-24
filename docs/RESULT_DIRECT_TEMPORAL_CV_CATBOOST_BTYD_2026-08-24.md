# RESULT — direct CatBoost plus BTYD ablation

- DataSphere job: `bt1o5qbmt3gttgvcqhhq`
- PRE-RUN commit: `ddcdbef85f447461d79a3ff86f4a080bb62aaad2`
- Config: `configs/direct_temporal_cv_v1/catboost_btyd_pending.yaml`
- Config SHA: `f131e05dff6331e13e5e005fb821676a9a335fff524583b5982dbdcf75646a43`
- `cv_summary.json` SHA-256: `1d537ca61b3e795f98c91c1e52ed8b900415b35db43e15bfd1479b6faf48fc2d`
- `sha256sums.json` SHA-256: `154684306c2b149eac1ced49c45829ef3d5186f0a93cf4401e9b9bf5265ddde9`

The only added columns were the causal classifier features
`btyd_p_buy_30d`, `btyd_expected_purchases_30d`, and `btyd_p_alive`. BG/NBD
parameters were fitted on each fold's train anchor and reused for that fold's
validation transform. No future target or Amount BTYD feature was exposed.

| Fold | Baseline RMSLE | + BTYD RMSLE | Delta |
|---|---:|---:|---:|
| F1 | 1.7031713232 | 1.7033619453 | +0.0001906221 |
| F2 | 1.7356890453 | 1.7357926573 | +0.0001036120 |
| F3 | 1.7435744432 | 1.7436581311 | +0.0000836879 |
| F4 | 1.6901233682 | 1.6902756549 | +0.0001522867 |
| **Mean** | **1.7181395450** | **1.7182720972** | **+0.0001325522** |

Decision: BTYD is not an improvement under this exact direct-CatBoost
four-fold protocol. Keep it as a recorded negative ablation; do not add it to
the ETT/TCN blend experiments.
