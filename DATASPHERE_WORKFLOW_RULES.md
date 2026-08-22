# Регламент разработки и запуска заданий на Yandex DataSphere (DATASPHERE_WORKFLOW_RULES.md)

---

## 🚨 Главный принцип: 100% локальная валидация перед отправкой на ВМ

**КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** отправлять в DataSphere непроверенный код «вслепую» в надежде, что он соберется в облаке.
Каждый цикл отсылки, ожидания очереди контейнера и сборки зависимостей на DataSphere занимает от 5 до 10 минут. Исправление опечаток по одной строчке через ВМ — грубейшее нарушение рабочего процесса.

---

## Обязательный пошаговый чеклист перед КАЖДЫМ запуском в DataSphere

### Шаг 1. Проверка синтаксиса и импортов (Static Analysis)
1. Выполнить компиляцию всех измененных скриптов:
   ```bash
   python -m py_compile scripts/your_script.py src/*.py
   ```
2. Проверить все импорты:
   * Не импортировать несуществующие функции (всегда проверять точное имя файла и функции в `src/`).
   * Не полагаться на память — делать `grep` или `view_file` для проверки сигнатур.

---

### Шаг 2. Обязательный локальный микро-прогон (Local Dry-Run)
Перед отправкой тяжелого скрипта на ВМ **ОБЯЗАТЕЛЬНО** запустить локальную проверку:
1. Запустить скрипт локально в режиме отладки (на 100–500 пользователях, 1 якоре или 5–10 итерациях моделей):
   * Проверить, что датасеты читаются без `FileNotFoundError`.
   * Проверить, что все колонки существуют в датафреймах (`polars.exceptions.ColumnNotFoundError`).
   * Проверить, что расчет метрик и объединение таблиц (`join`, `hstack`) не падает с ошибками размерностей.
   * Проверить запись выходных артефактов в директорию `artifacts/`.

---

### Шаг 3. Инвариантность путей к файлам (Container-Safe Path Resolution)
В контейнере DataSphere корень проекта монтируется как `/job/`. Поэтому:
1. **Никогда** не зашивать жесткие абсолютные пути вида `Path(__file__).parents[1] / "data" / ...`.
2. **Всегда** использовать защищенное определение путей:
   ```python
   DATA_DIR = Path("data") if Path("data").exists() else Path(".")
   SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
   TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
   USERS_PARQUET = (
       Path("artifacts/selected_users_100k.parquet")
       if Path("artifacts/selected_users_100k.parquet").exists()
       else (Path("selected_users_100k.parquet") if Path("selected_users_100k.parquet").exists() else Path("artifacts/selected_users_100k.parquet"))
   )
   OUTPUT_DIR = Path("artifacts/your_experiment")
   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
   ```

---

### Шаг 4. Проверка YAML-конфигурации DataSphere
1. Все пути, указанные в `local-paths:`, **обязаны существовать на локальном диске** перед запуском `datasphere_runner.py` (иначе CLI DataSphere упадет на этапе упаковки zip-архива).
2. Выходная папка из `outputs:` должна быть создана локально до вызова CLI:
   ```python
   Path("artifacts/your_experiment").mkdir(parents=True, exist_ok=True)
   ```
3. Все необходимые сторонние пакеты (`lifetimes`, `catboost`, `lightgbm`, `polars`) должны быть зафиксированы в `requirements-datasphere.txt`.

---

### Шаг 5. 🚨 Сборка тяжелых датасетов и признаков ТОЛЬКО на ВМ
1. **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** загружать на ВМ тяжелые директории с сотнями сгенерированных колонок (например, `artifacts/specialized_hurdle/feature_store/`, `oof/`).
2. В `local-paths:` передаются только исходные файлы (`data/train.parquet`, `data/snapshots/`, `selected_users_100k.parquet`), код и конфиги.
3. Вся сборка обучающих выборок, объединение по якорям (`pooled dataset`) и фильтрация признаков **ОБЯЗАНЫ выполняться на лету на самой ВМ в начале исполнения скрипта**.

---

## Чеклист готовности к отправке:
- [ ] Все модули `src/` и `scripts/` компилируются без ошибок синтаксиса.
- [ ] Все имена функций и колонок датафреймов сверены с реальными файлами.
- [ ] Локальный микро-тест прошел успешно за < 10 секунд.
- [ ] Директории вывода созданы локально.
- [ ] Никаких тяжелых промежуточных сгенерированных таблиц нет в `local-paths:`.
- [ ] Пути в скрипте адаптированы под запуск как локально, так и в `/job/`.
- [ ] Только ПОСЛЕ этого запускается `scripts/datasphere_runner.py`.
