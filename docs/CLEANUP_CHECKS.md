# Portfolio cleanup checks — 2026-10-03

Проверки выполнены в существующем conda myenv, из корня, с PYTHONPATH=.
Обучение, full preprocessing/inference и cloud jobs не выполнялись.

| Проверка | Результат |
|---|---|
| `pytest tests/direct_temporal_cv_v1 tests/test_datasphere_runner.py -q -p no:cacheprovider` | 15 passed, 1 failed |
| CV `--help`, baseline `--contract-check` | Passed |
| Quad builder / CoLES209 builder / runner `--help` | Passed |
| YAML parse `configs/datasphere/**/*.yaml` | 90 passed |
| Entry point paths из cmd перенесённых manifests | Все существуют |
| `py_compile` relocated research и изменённых Python modules | 45 passed |
| Markdown links новых README/index/reproduction/provenance | Все существуют |
| `git diff --check` | Passed |

Единственный оставшийся test failure:
`test_datasets_features.py::test_features_are_causal_and_sparse`. Его существующий
8-column fixture не содержит `cat`, `cat_to_cart`, `cat_to_ord`,
`search_to_cart`, `search_to_ord`, требуемые более поздним feature provider.
Переносы не изменили математику features. Новые fixtures/tests не создавались.
В существующих runner tests изменены только пути YAML.

Исторические manifests: 22 дублируют inputs/local-paths; 97 ссылок на восемь
отсутствующих локальных входов (один и тот же вход указан в нескольких YAML):

- `data/snapshots/`
- `artifacts/selected_users_100k.parquet`
- `selected_users_100k.parquet`
- `artifacts/reference_v1/cohorts/POST_NY_PUBLIC_PROXY/full_cohort_100k.parquet`
- `artifacts/reference_v1/cohorts/POST_NY_PUBLIC_PROXY/screen_cohort_25k.parquet`
- `models/gru_sweep/gru_len_L180_recent14/best.pt`
- `test_specialists_raw_predictions_250k.parquet`
- `test_specialists_raw_predictions_250k_v2.parquet`

Они сохранены как historical recipes, а не объявлены готовыми к запуску.
Raw `data/train.parquet` и `sample_submit.csv` доступны локально, но вне Git.
Финальное team submission не воспроизводилось; точный CSV/score lineage
описан в [SUBMISSION_PROVENANCE.md](SUBMISSION_PROVENANCE.md).

До начала работ worktree уже содержал изменения `.agents/AGENTS.md`,
`DATASPHERE_AGENT_RUNBOOK.md`, `scripts/datasphere_cli_wrapper.py` и untracked
AGENTS/два final manifests/config/CB+ETT builder. Они сохранены. В инструкциях
и ссылках скорректированы пути; wrapper и пользовательские experiment recipes
не переписаны. Commit/push не выполнялись.
