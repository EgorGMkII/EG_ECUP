# Инструкция по запуску скриптов в Yandex DataSphere

DataSphere удобно использовать как “удаленный запуск скрипта по YAML-конфигу”. Общая схема такая.

**1. Один раз подготовить окружение локально**
В каждом проекте лучше иметь отдельный venv:

```powershell
python -m venv .venv-ds
.\.venv-ds\Scripts\Activate.ps1
python -m pip install -U pip
pip install datasphere
```

Проверка:

```powershell
datasphere -h
```

Если предупреждение про YC initialization мешает:

```powershell
$env:YC_CLI_INITIALIZATION_SILENCE = "true"
```

Навсегда для пользователя:

```powershell
[Environment]::SetEnvironmentVariable("YC_CLI_INITIALIZATION_SILENCE", "true", "User")
```

**2. Авторизация**
Лучше не вставлять токен в команды каждый раз. На текущую PowerShell-сессию:

```powershell
$env:YC_TOKEN = "<твой_yandex_oauth_token>"
```

Потом команды можно запускать так:

```powershell
datasphere -t $env:YC_TOKEN project job list -p <PROJECT_ID>
```

Токены не коммить, не класть в YAML, не отправлять в чат.

**3. Минимальный `datasphere.train.yaml`**
Пример для любого Python-проекта:

```yaml
name: my-train-job
desc: Run training script

cmd: >
  python3 train.py

env:
  python:
    type: manual
    version: 3.10.13
    requirements-file: requirements-datasphere.txt
    local-paths:
      - src/
      - train.py
      - config.yaml
      - data/input.csv

outputs:
  - outputs/
  - artifacts/

cloud-instance-types:
  - g1.1

working-storage:
  type: SSD
  size: 100Gb

graceful-shutdown:
  signal: SIGTERM
  timeout: 30s
```

Главные поля:

- `cmd`: что запустить в облаке.
- `requirements-file`: зависимости, которые поставить в job environment.
- `local-paths`: какие локальные файлы/папки отправить в job snapshot.
- `outputs`: какие папки/файлы потом можно скачать.
- `cloud-instance-types`: тип железа, например GPU `g1.1`.

**4. Запуск**
Из корня проекта:

```powershell
datasphere -t $env:YC_TOKEN project job execute -p <PROJECT_ID> -c datasphere.train.yaml
```

Посмотреть список job:

```powershell
datasphere -t $env:YC_TOKEN project job list -p <PROJECT_ID>
```

Скачать результаты:

```powershell
datasphere -t $env:YC_TOKEN project job download-files --id <JOB_ID>
```

**5. Практические правила**
- Все файлы, которые скрипт читает, должны быть в `local-paths`, либо лежать в облачном хранилище.
- Все файлы, которые хочешь забрать после запуска, должны попадать в `outputs`.
- Пути внутри job обычно начинаются от `/job/`.
- Локальный Windows path типа `C:\...` внутри job не существует.
- Для больших датасетов лучше не гонять CSV каждый раз, а использовать Object Storage/S3 или кеш DataSphere jobs.
- `wandb` лучше настраивать через переменную окружения/секрет в облаке, а не через локальный `wandb login`.

Официальная документация: DataSphere Jobs запускаются через `datasphere project job execute -p <project_ID> -c <configuration_file>`, CLI ставится через `pip install datasphere`, поддерживаются Python venv и версии Python 3.8-3.12. Источники: [Yandex DataSphere Jobs](https://yandex.cloud/ru/docs/datasphere/operations/projects/work-with-jobs), [DataSphere CLI](https://yandex.cloud/en/docs/datasphere/concepts/jobs/cli).
