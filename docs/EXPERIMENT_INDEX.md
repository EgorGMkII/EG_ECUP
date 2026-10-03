# Индекс экспериментов

| Направление | Отчёты | Код |
|---|---|---|
| Direct baseline CV | `results/RESULT_DIRECT_TEMPORAL_CV_CATBOOST_BASELINE_2026-08-24.md` | `src/direct_temporal_cv_v1/` |
| BTYD / leakage | `reports/BTYD_LEAKAGE_AUDIT.md`, `results/RESULT_DIRECT_TEMPORAL_CV_CATBOOST_BTYD_2026-08-24.md` | `src/btyd_pipeline.py`, direct `btyd.py` |
| ETT/TCN/LightGBM/CoLES | `results/RESULT_DIRECT_CV_NEXT_GEN_PACK_V1.md`, локальные CV manifests | direct adapters, `coles.py` |
| Поздние submission stacks | `SUBMISSION_PROVENANCE.md`, `results/RESULT_SUBMISSION_NEXT_GEN_QUAD_V1.md` | `scripts/build_direct_*submission.py` |
| SSL/RUN A→B | `SSL_TEMPORAL_STACK_V1_SPEC.md`, `results/RESULT_SSL_TEMPORAL_STACK_V1_2026-08-23.md` | `src/ssl_temporal_stack_v1/`, `src/reference_framework_v1/` |
| Hurdle/GRU/transition | `reports/EDA_RESULTS.md`, `reports/SPECIALIZED_HURDLE_FLOW.md` | `src/sequential/`, `src/transitions/` |
| Ранние exploratory scripts | `../archive/scratch/README.md` | `archive/scratch/` |

Исторические отчёты содержат наблюдения конкретных запусков. Рекомендации и
ожидания leaderboard в них не являются подтверждёнными результатами.
Offline score читать вместе с cohort, folds, feature manifest и SHA.

Ноутбуки в корне: `baseline-seacrh-ltv.ipynb` — начальная разведка;
`eda.ipynb` — EDA; `train_classifier.ipynb`, `train_hurdle.ipynb` — табличные;
`train_sequential.ipynb` — последовательности; `transition_modeling.ipynb` —
переходы; `gru_sweep_analysis.ipynb` — sweep. Требуют локальные artifacts.
