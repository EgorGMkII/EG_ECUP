# Воспроизведение и модули

Все команды — из корня, `myenv`, `PYTHONPATH=.`. Данные не скачиваются автоматически.

1. `data/train.parquet` + `sample_submit.csv`: sparse daily log и ordered users.
2. `src/direct_temporal_cv_v1/datasets.py`: target, log1p, template alignment.
3. `features.py`: causal sparse windows 7/14/30/60/90/180/365, GMV и активности,
   recency/history, поздние funnel/catalog признаки. `btyd.py`: BTYD extension.
4. `src/ssl_temporal_stack_v1/stores.py`: daily float32/event FP16 memmaps.
   `src/direct_temporal_cv_v1/coles.py`: self-supervised embeddings.
5. Direct `adapters/*.py`: обучение/inference; `registry.py`: IDs;
   `pipeline.py`: независимые folds, banks, reports/hashes.
6. `scripts/build_direct_*submission.py`: fresh full training, inference,
   fixed blend weights, expm1/clipping, CSV.

ETT architecture — `src/ssl_temporal_stack_v1/models.py`; TCN blocks —
`src/reference_framework_v1/candidates/tcn.py`. Direct training loops —
`src/direct_temporal_cv_v1/adapters/direct_ett.py` и `direct_tcn.py`.

## Быстрые проверки

```powershell
conda activate myenv
$env:PYTHONPATH = '.'
python scripts/run_direct_temporal_cv.py --help
python scripts/run_direct_temporal_cv.py --experiment-config configs/direct_temporal_cv_v1/baseline_catboost.yaml --contract-check
python scripts/build_direct_next_generation_submission.py --help
python scripts/build_direct_coles209_triple_stack_submission.py --help
python scripts/datasphere_runner.py --help
python -m pytest tests/direct_temporal_cv_v1 tests/test_datasphere_runner.py -q -p no:cacheprovider
```

## Дорогие recipes

CV: тот же entrypoint без `--contract-check`, с `--pre-run-sha <SHA>`.
Новый YAML ID/output root обязателен при существующих результатах.

Final triple: `scripts/build_direct_coles209_triple_stack_submission.py
--pre-run-sha <SHA> --output-root artifacts/<new_run>`.
Multi-anchor: `scripts/build_direct_multi_anchor_750k_triple_stack_submission.py`
с теми же аргументами. Это разные recipes.

Quad: `scripts/build_direct_next_generation_submission.py --pre-run-sha <SHA>
--output-dir artifacts/<new_run>`. Для дополнительного blend нужен корневой
`submission_meta_blend_champions_v1.csv`. Builder допускает существующую папку;
новую выбирать вручную. Этот recipe не гарантирует командный score.

Тяжёлые этапы выполнять через DataSphere runner, manifests в `configs/datasphere/`.
[Правила](../DATASPHERE_AGENT_RUNBOOK.md). Исторические YAML сохраняют старый
transfer layout; часть дублирует raw paths в inputs/local-paths. Перед новым
запуском требуется аудит, archive не означает готовность к запуску.

ETT event snapshot 250k × 180 ~2.3 GB диска, daily float32 ~2.7 GB. Multi-anchor
`np.concatenate` создаёт копии в RAM. GPU V100 g1.1, SSD 100 GB использовались
исторически; peak RAM не измерен. Данные/weights/banks/submissions вне Git.
Пробелы финального lineage: [provenance](SUBMISSION_PROVENANCE.md).

Ноутбуки оставлены в корне для cwd-relative data/artifact paths. Они являются
исследованиями, не обязательными шагами final recipe.
