# RESULT — direct CatBoost four-fold 250k baseline

## Provenance

- DataSphere job: `bt1grsllg79uo3bnaoh9`
- PRE-RUN commit: `8ee9ed39839a49e2e53dcad7f4fa6da7bdf78746`
- Protocol: `FOUR_FOLD_250K_V1`
- Config: `configs/direct_temporal_cv_v1/baseline_catboost.yaml`
- Config SHA: `9de10adfcf9af7ad0e2902fc75aa36c378e46016d03cf49c94882eabc63bea0f`
- Output root: `artifacts/direct_temporal_cv_v1/experiments/direct_cv_catboost_baseline_v1/`
- `cv_summary.json` SHA-256: `8e43830e9d352268ffa2d58e48c8a8133b802386baf9e169339d80bd41cf27a9`
- `sha256sums.json` SHA-256: `ced5a5c2210b1a38f5da566081ac7050c545c9e9a5a68ff2674fca0045344416`

## Recipe

Direct `CatBoostRegressor` on all 250,000 template users, sparse causal
features, no random split, no hurdle/meta, and one fresh model per fold:

`iterations=300`, `depth=8`, `learning_rate=0.05`, `l2_leaf_reg=5`,
`loss_function=RMSE`, `random_seed=42`, `thread_count=8`.

## Validation result

| Fold | Inference anchor | RMSLE |
|---|---|---:|
| F1 | 2025-10-16 | 1.7031713232 |
| F2 | 2025-11-15 | 1.7356890453 |
| F3 | 2025-12-15 | 1.7435744432 |
| F4 | 2026-01-14 | 1.6901233682 |
| **Mean** | — | **1.7181395450** |

The reported external reference `1.717960` is reproduced within
`+0.0001795450` RMSLE. This baseline is now the immutable comparison point for
BTYD, direct ETT, direct TCN, and later leakage-safe blends.
