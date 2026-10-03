# Рабочий контракт: direct temporal CV / DataSphere

Перед любым изменением или запуском прочитать:

1. `DATASPHERE_AGENT_RUNBOOK.md` — единственный источник точного способа
   отправки и мониторинга DataSphere job.
2. `DATASPHERE_WORKFLOW_RULES.md` — pre-flight, пути `/job`, outputs и
   каузальность данных.
3. `docs/` RESULT-документы и `artifacts/direct_temporal_cv_v1/...` —
   фактические метрики и hashes уже завершённых экспериментов.

## Текущая цель

Порядок экспериментов неизменен: direct CatBoost baseline/BTYD → direct ETT
→ direct TCN → leakage-safe blend трёх моделей → full 250k training →
отдельный submission. Не подменять этот протокол старыми hurdle/public
entrypoints и не смешивать их artifacts.

## Код и эксперименты

- Работать в `myenv`.
- Один experiment YAML описывает ровно один immutable run и один уникальный
  output root. Даты/folds берутся из versioned protocol, а не копируются в
  скрипты.
- Валидация: четыре временных fold на 250k. Любое мета-решение обучается
  только на разрешённых ранних fold; holdout fold никогда не используется для
  подбора модели, checkpoint или веса.
- Каждый model report обязан содержать RMSLE/MSE, fold metrics, prediction
  hash, transition `00/01/10/11`, GMV buckets, параметры и seed.
- Модели в full run обучаются с нуля. Единственный переносимый объект между
  этапами — явно зафиксированный meta package/weights.
- Не перезаписывать submissions, prediction banks, checkpoints и RESULT
  artifacts. Новый run — новый output root.

## DataSphere — обязательный способ

- Запускать только:
  `C:\Users\egorg\anaconda3\envs\myenv\python.exe scripts/datasphere_runner.py -c <manifest> --pre-run-sha <SHA>`.
- Строго синхронно; `--async` и прямой вызов CLI запрещены.
- Manifest layout: `local-paths` содержит только `src/`, `scripts/`,
  `configs/`, `requirements-datasphere.txt`; raw `data/train.parquet` и
  `sample_submit.csv` находятся только в `inputs`. Не дублировать paths.
- Временные CLI module archives создаёт `scripts/datasphere_cli_wrapper.py` в
  `.datasphere_tmp/`: Windows endpoint блокирует запись в новые `%TEMP%`
  directories. Не заменять этот wrapper и не менять transfer layout без
  успешного `configs/datasphere/smoke/datasphere.smoke.yaml`.
- Heavy feature stores, event/daily memmap, frames, caches и checkpoints
  строятся только на VM; они не входят ни в `local-paths`, ни в `outputs`.
- В `outputs` указывать только конечные small artifacts: reports, manifests,
  prediction banks, meta package, submission. Не выгружать рекурсивный
  `artifacts/` с transient stores.
- Мониторинг существующего job: runner `--id <JOB_ID>` не чаще одного раза в
  минуту. Не использовать attach/restart и не создавать второй job, пока
  предыдущий того же назначения не terminal.

## Provenance

- До тяжёлого run: py_compile, YAML parse, micro dry-run, input/output
  inspection, PRE-RUN commit и обычный push.
- После terminal success: schema/finite/alignment checks, hashes, RESULT
  commit с job ID, duration, metrics и artifact hashes.
- Если Git write заблокирован локальной sandbox/ACL, не подменять commit:
  отметить блокер, сохранить exact training SHA и восстановить commit до
  следующего экспериментального запуска.
