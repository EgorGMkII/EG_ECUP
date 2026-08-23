# DataSphere: рабочий регламент агента

Этот файл — короткая операционная инструкция для запуска и сопровождения
одного эксперимента. Главные источники правил: `DATASPHERE_WORKFLOW_RULES.md`,
`DATASPHERE.md` и `SPECIALIZED_HURDLE_FLOW.md`. При противоречии приоритет у
регламента workflow и фактической упаковки текущей версии CLI.

## Непереговорные правила

- Запускать только из `myenv` и только через
  `scripts/datasphere_runner.py`; прямой вызов DataSphere CLI не использовать.
- Не запускать второй экземпляр, пока предыдущий job с тем же назначением не
  перешёл в terminal state.
- Не менять код, конфиг или объявленные outputs работающего job. Исправление
  оформляется новым PRE-RUN commit и новым job только после явного анализа
  terminal failure.
- Не передавать готовые feature stores, OOF-банки, checkpoints или прочие
  тяжёлые производные витрины. В job отправляются только исходные данные,
  код и конфигурации; витрина строится на GPU VM.
- Не объявлять каталог с checkpoint/cache в `outputs`. Выгружать только
  маленькие конечные файлы: manifest, мета-веса, prediction bank и report.
- Для `reference_framework_v1` сначала построить и проверить immutable
  cohort manifests; каждый DataSphere job получает один resolved experiment
  config и один run-scoped output root. Sweep YAML сам jobs не запускает.

## Gate до создания job

Все пункты обязательны и выполняются в этом порядке.

1. Прочитать три документа выше и текущие `AGENTS.md` правила.
2. Зафиксировать ровно один PRE-RUN commit с кодом и YAML, затем обычный
   `git push`. SHA передаётся job через environment variable: внутри `/job`
   нет каталога `.git`, поэтому `git rev-parse` там запрещён.
3. Проверить все изменённые Python-файлы через `py_compile`, затем запустить
   локальный micro dry-run на 100–500 пользователей. Он обязан проверять пути,
   схему, метрики и запись конечных outputs.
4. Проверить YAML: каждый исходный путь существует; requirements фиксируют
   нужные зависимости; `cmd` содержит `export PYTHONPATH=.`; все пути в коде
   относительны `/job` (`data/train.parquet`, `scripts/...`, `configs/...`).
5. Проверить layout архива, а не гадать по YAML. Код передаётся каталогами
   (`src/`, `scripts/`, `configs/`), чтобы в контейнере существовали
   `/job/src/...` и `/job/scripts/...`; одиночный файл может попасть в корень.
   Набор данных проверяется по фактическому upload-логу: размер
   `data/train.parquet` должен соответствовать исходному файлу.
6. Вычислить бюджет результатов до запуска. Каждый `outputs:` — явный
   конечный файл, не рекурсивный `artifacts/` каталог. Caches, memmap и
   checkpoints остаются рабочими файлами VM и очищаются самим скриптом.
7. Запустить один job командой вида:

   ```powershell
   & 'C:\Users\egorg\anaconda3\envs\myenv\python.exe' scripts\datasphere_runner.py -c datasphere.<experiment>.yaml
   ```

   Новые job запускаются runner-ом в синхронном streaming mode. Не добавлять
   `--async`: CLI должен в реальном времени обновлять локальные `stdout.txt`,
   `stderr.txt`, `gpu_stats.tsv` и `docker_stats.tsv`. Сам job-скрипт обязан
   печатать stage/step heartbeat через `print(..., flush=True)` — нужны оба
   условия.

8. Немедленно записать job ID, PRE-RUN SHA, config SHA256 и ожидаемые outputs
   в журнал эксперимента.

## Наблюдение без повторного запуска

Для уже созданного job допустима только status-команда:

```powershell
& 'C:\Users\egorg\anaconda3\envs\myenv\python.exe' scripts\datasphere_runner.py --id <JOB_ID>
```

Она делает один запрос `project job get` и не создаёт обучение. Polling — это
не pooling: он не использует GPU и не запускает модели. Проверять статус не
чаще раза в 60 секунд, не плодить параллельные polling-команды.

`--attach` не использовать для мониторинга: в используемой версии CLI он
может инициировать повторное исполнение или вести себя неоднозначно. Также не
пытаться скачать outputs, пока job не завершён — частичные outputs не являются
достоверным результатом.

Для тяжёлого multi-anchor pipeline заранее фиксируется полный union anchors.
На VM один раз строятся causal feature frames и pooled sequence stores; RUN A и
RUN B получают только разрешённые им anchor slices. Совпадающий causal frame
переиспользуется, но модели RUN B всегда создаются и обучаются с нуля.

Последовательности ETT хранить как pooled memmap (`float16` content/time,
`int16` ranks, `bool` masks), без `np.savez_compressed`. Cache hit обязан
проверяться до фильтрации raw log; multi-horizon labels вычисляются один раз на
anchor, а не заново в каждом epoch.

Локальная ошибка формирования команды, например `SyntaxError: Unexpected end
of input` до запуска PowerShell, означает сбой локальной оркестрации, а не
ошибку DataSphere job. Это проверяется следующим status-запросом по job ID.

## Действия в terminal state

### COMPLETED

1. Скачать только объявленные outputs через runner.
2. Проверить JSON/Parquet: schema, число строк, finite prediction, даты
   anchors, метрики и хеши файлов.
3. Не перезаписывать submission или воспроизводимый artifact.
4. Создать RESULT commit с job ID, PRE-RUN SHA, RMSLE/MSE, transition и
   React/Churn/Amount diagnostics, SHA256 config и artifact hashes; затем
   обычный `git push`.

### ERROR

1. Зафиксировать точный terminal status и доступный stderr, без догадок.
2. Классифицировать проблему: packaging/path, отсутствие `.git` в `/job`,
   runtime/schema, dependency или превышение output budget.
3. Исправить локально, снова пройти весь Gate, сделать новый PRE-RUN commit и
   лишь затем создать следующий job. Никаких restart/attach «на удачу».

## Известные анти-паттерны этого проекта

- Одиночный `scripts/foo.py` в package вместо каталога `scripts/` — риск
  отсутствия `/job/scripts/foo.py`.
- `git rev-parse HEAD` внутри job — `/job` не является Git checkout.
- `outputs: artifacts/.../` — может выгрузить cache/checkpoint и превысить
  лимит размера результата.
- Обучение по модели на каждый фолд вместо pooled training по anchors — не
  соответствует каноническому Specialized Hurdle flow и создаёт ложную оценку
  длительности.
