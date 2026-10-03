# Ozon E-Cup 2026 — прогноз GMV на 30 дней

Соревновательный ML-проект: предсказать суммарный GMV за следующие 30 дней для
250 000 пользователей по разреженным дневным логам Ozon. Здесь сохранены
признаки на Polars, временная валидация, табличные и последовательностные модели,
диагностика ошибок и несколько поколений ансамблей.

**Итог команды: RMSLE 1.6558615631, место 152 из 316.** Результат предоставлен
автором; [leaderboard соревнования](https://ods.ai/tracks/e-cup-2026-competitions/competitions/e-cup-2026-search/leaderboard).
Это leaderboard-метрика. Связь этого score с конкретным загруженным CSV и его
SHA-256 в сохранённых отчётах отсутствует.

## Решение и результаты

Последнее документированное обучение — Next-Generation Quad Stack:
`scripts/build_direct_next_generation_submission.py`, job `bt1n2e9pa3592plu15bk`.
Он обучает direct CatBoost, частотную пару CatBoost/LightGBM, CatBoost на остатках
GMV и Two-Tower сеть, используя sparse-признаки и 32 CoLES embeddings. Затем
собирает ансамбль в log-space и его blend (40%) с ранее сохранённым
`submission_meta_blend_champions_v1.csv` (60%).

Есть также builders CatBoost + LightGBM + multi-task Event-Time Transformer
(ETT), включая три snapshots × 250k = 750k обучающих строк. Код, manifests
и CSV подтверждают существование этих решений, но не позволяют назначить
какому-либо из них итоговый командный score. Более ранний CB+ETT с Public
RMSLE 1.6572245219 — отдельный результат. Подробности:
[submission provenance](docs/SUBMISSION_PROVENANCE.md).

| Эксперимент | Offline RMSLE | Выборка |
|---|---:|---|
| Direct CatBoost, первоначальный запуск | 1.7181395450 | Среднее F1–F4, все 250k |
| Direct ETT, первоначальный запуск | 1.7298917377 | Среднее F1–F4 |
| Direct TCN, первоначальный запуск | 1.7736966448 | Среднее F1–F4 |
| CatBoost, первоначальный запуск | 1.6901233682 | F4 |
| CB+ETT, веса fit на F1–F3 | 1.6872863347 | F4 |
| Next-Gen Quad Stack, отчёт запуска | 1.686582 | F4, веса fit на F1–F3 |

Это результаты конкретных исторических конфигураций. Текущий feature provider
развивался после первого baseline: исходный manifest фиксировал 68 признаков,
поздние версии добавили funnel/catalog признаки и CoLES. Старый YAML на текущем
коде не гарантирует повторение исходной метрики.

## Временная валидация

`src/direct_temporal_cv_v1/contracts.py` задаёт cutoff T: 2025-10-16,
2025-11-15, 2025-12-15, 2026-01-14. В каждом fold модель создаётся с нуля:
признаки на T−30, target (T−30,T], затем признаки на T и target (T,T+30].
Все 250k пользователей сохраняются в порядке template; отсутствующий будущий
GMV равен нулю. Random split не используется. Веса blend подбирались на
F1–F3; F4 служил отдельной проверкой.

RMSLE = sqrt(mean((log1p(prediction) − log1p(target))²)). Direct-модели учатся
на `log1p(GMV)`; submission получает `max(expm1(z),0)`. Offline и Public score
относятся к разным выборкам.

## Данные и воспроизведение

Нужны файлы соревнования `data/train.parquet` и `sample_submit.csv` с 250k
уникальными `user_id` в требуемом порядке. Они не включены в Git. История:
2025-01-01…2026-02-13, около 30 млн наблюдаемых дневных записей. Колонки:
`user_id`, `event_date`, `gmv`, `gmv_search`, `gmv_cat`, `searches`, `to_ord`,
`to_cart`, `cat`, `search_to_ord`, `cat_to_ord`, `search_to_cart`, `cat_to_cart`.

Маршрут: raw parquet → causal snapshots/последовательности → fresh training
→ inference на 2026-02-13 → log-space blend → CSV `user_id,predict`.
Single-snapshot final учится на признаках 2026-01-14 и target
2026-01-15…2026-02-13; прогнозный месяц — 2026-02-14…2026-03-15.

Стек: Python, NumPy, **Polars**, PyArrow, SciPy, scikit-learn, CatBoost,
LightGBM, PyTorch, PyYAML; часть BTYD экспериментов использует lifetimes.
Cloud environment: Python 3.10.13, PyTorch 2.3.1/CUDA 12.1 в
`requirements-datasphere.txt`; локально использовалась conda `myenv`.
Полного lockfile нет, часть зависимостей задана нижними границами версий.

Из корня репозитория, быстрые проверки:

```powershell
conda activate myenv
$env:PYTHONPATH = '.'
python scripts/run_direct_temporal_cv.py --help
python scripts/run_direct_temporal_cv.py --experiment-config configs/direct_temporal_cv_v1/baseline_catboost.yaml --contract-check
python -m pytest tests/direct_temporal_cv_v1 -q -p no:cacheprovider
```

`--contract-check` не читает полный parquet и не обучает модели. Полный CV —
тот же entrypoint без этого флага, с `--pre-run-sha <SHA>`; это дорогой запуск.
Облачные jobs отправляются только через runner, например поздний triple stack:

```powershell
python scripts/datasphere_runner.py -c configs/datasphere/submissions/datasphere.submission_coles209_triple_stack.yaml --pre-run-sha <SHA>
python scripts/datasphere_runner.py --id <JOB_ID>
```

Quad Stack manifest:
`configs/datasphere/submissions/datasphere.direct_next_generation_submission.yaml`.
Для дополнительного champion blend нужен `submission_meta_blend_champions_v1.csv`.
Без него builder создаёт только Quad Stack. Перед запуском выбрать новый output
root и пройти [DataSphere pre-flight](DATASPHERE_AGENT_RUNBOOK.md).

ETT/CoLES/TCN рассчитаны на CUDA; исторически использовался Tesla V100 (`g1.1`).
Raw parquet читается в RAM. Один daily tensor `[250000,180,15]` float32 занимает
~2.7 GB, один event snapshot — ~2.3 GB на диске. Планировать GPU класса V100
16 GB, десятки GB RAM и до 100 GB SSD; фактический пик RAM не зафиксирован.
Multi-anchor builder конкатенирует memmaps в RAM. Эти этапы здесь не воспроизводились.

## Навигация

| Путь | Назначение |
|---|---|
| `src/direct_temporal_cv_v1/` | Direct CV: данные, признаки, adapters, обучение, метрики |
| `src/ssl_temporal_stack_v1/` | Переиспользуемые ETT/stores и прежний SSL specialist flow |
| `src/reference_framework_v1/` | Исторический RUN A/B framework, TCN/MLP |
| `src/sequential/`, `src/transitions/`, `src/specialized_hurdle/` | GRU/Transformer и hurdle исследования |
| `scripts/run_direct_temporal_cv.py` | Общий CV entrypoint |
| `scripts/build_direct_*submission.py` | Отдельные final-training recipes |
| `configs/direct_temporal_cv_v1/` | Конфиги direct экспериментов |
| `configs/datasphere/{cv,submissions,smoke,archive}/` | Облачные manifests |
| `docs/results/`, `docs/reports/` | Результаты запусков и исследования |
| `archive/scratch/`, `scripts/research/` | Исторические скрипты и cloud probe |
| `artifacts/`, `data/`, корневые CSV/weights | Локальные артефакты, вне Git |
| Корневые `.ipynb` | EDA и исследования; запуск из корня |

Подходы: direct boosting, hurdle/React-Churn-Amount specialists, GRU,
ETT/Transformer, causal TCN, Residual MLP, BTYD и CoLES. TCN/GRU/MLP исследовались
отдельно; их участие в финальном командном CSV не установлено.
[Индекс экспериментов](docs/EXPERIMENT_INDEX.md) ·
[Модули и команды](docs/REPRODUCTION.md) · [Карта переносов](docs/FILE_RELOCATIONS.md).
