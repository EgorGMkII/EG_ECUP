# Документация по предобработке данных в Polars (`src/data.py`)

Данный документ содержит полное руководство по архитектуре, методам и нативным оптимизациям **Polars**, используемым для предобработки данных и расчета признаков в соревновании Ozon LTV.

---

## 1. Архитектура и стратегия обработки данных

Основная задача предобработки — преобразовать исходные дневные логи кликов/покупок (`train.parquet`, 30.6 млн строк) в табличный формат `(anchor_date, user_id)` с временными агрегатами (фичами) и LTV-таргетом на 30 дней вперёд.

### Ключевые концепции:
1. **Временной сплит по якорным датам (Time-CV Anchors)**:
   - Временной ряд разбивается на исторические срезы на якорные даты (`anchor_date`).
   - Для каждого якоря фичи рассчитываются **только по историческим дням** ($\le \text{anchor\_date}$), а таргет — по **следующим 30 дням** ($[\text{anchor\_date} + 1 \dots \text{anchor\_date} + 30]$).
2. **Бачевание по пользователям (User Batching)**:
   - Все 250 000 пользователей делятся на батчи по 50 000 пользователей (`BATCH_SIZE = 50_000`).
   - Это гарантирует нулевой риск переполнения оперативной памяти (RAM) при выполнении тяжёлых группировок `group_by`.

---

## 2. Нативные методы и концепции Polars

### 2.1. Expression API (`pl.Expr`)
В отличие от Pandas, Polars не выполняет промежуточные операции сразу в памяти. Вместо этого формируются **выражения (`pl.Expr`)**, которые оптимизируются нативно на языке Rust.

Пример генерации агрегатов с маской времени (`pl.when().then().otherwise()`):
```python
# Фильтр временного окна прямо внутри агрегации
mask = pl.col("event_date").is_between(w_start, w_end)

# Нативное выражение Polars без создания промежуточных копий датафрейма
sum_expr = pl.when(mask).then(pl.col("gmv")).otherwise(0.0).sum()
```

### 2.2. Извлечение признаков с `group_by` и `agg`
Массив выражений передаётся в метод `.agg()`, где Polars параллельно по всем ядрам процессора выполняет векторные вычисления:

```python
features = (
    df_filtered.group_by("user_id")
    .agg(build_window_agg_exprs(anchor_date, windows, value_cols, aggs))
    .with_columns(anchor_date=pl.lit(anchor_date))
)
```

### 2.3. Посегментный фильтр (`is_between` и `is_in`)
Перед агрегацией данные сначала отсекаются по границам нужных дат и списку пользователей:

```python
data_filtered = data.filter(
    pl.col("user_id").is_in(user_ids)
    & pl.col("event_date").is_between(min_date, max_date)
)
```

### 2.4. Заполнение пропусков (`fill_null`)
Для пользователей, не имевших активности в конкретном окне, результаты агрегата автоматически приводятся к `0.0`:

```python
fill_exprs = [pl.col(col).fill_null(0.0) for col in feature_cols]
result = result.with_columns(fill_exprs)
```

### 2.5. Мульти-файловое считывание Parquet (`read_parquet` с глобом)
Polars умеет считывать целые директории parquet-файлов параллельно, автоматически склеивая строки:

```python
# Загрузка всех 5 батчей фолда в один датафрейм за доли секунды
fold_df = pl.read_parquet("data/v2/features/fold_03/batch_*.parquet")
```

---

## 3. Справочник функций модуля `src/data.py`

### `generate_cv_anchor_dates(data, prediction_horizon_days=30, stride_days=14, min_history_days=90, n_folds=4)`
Генерирует список якорных дат для валидационных фолдов со сдвигом `stride_days`.
* **Возвращает**: `list[date]`

### `build_window_agg_exprs(anchor_val, windows, value_cols, aggs)`
Строит список выражений `pl.Expr` для временных окон и агрегатов (`sum`, `max`, `std`, `mean`).

Поддерживаемые типы окон в `DEFAULT_WINDOWS`:
* **Кумулятивные окна (накопительные)**:
  * `30d`: последние 30 дней (`[t-29 ... t]`)
  * `60d`: последние 60 дней (`[t-59 ... t]`)
  * `90d`: последние 90 дней (`[t-89 ... t]`)
* **Лаговые бакеты (непересекающиеся интервалы)**:
  * `30_60d`: период от 30 до 59 дней назад (`[t-59 ... t-30]`)
  * `60_90d`: период от 60 до 89 дней назад (`[t-89 ... t-60]`)

### `generate_features(data, anchor_dates, user_ids, value_cols, windows, aggs)`
Формирует датафрейм признаков `(anchor_date, user_id, gmv_sum_30d, gmv_sum_60d, gmv_sum_30_60d, ...)` для выбранных пользователей и якорных дат.

### `generate_targets(data, anchor_dates, user_ids, horizon_days=30, target_col="gmv")`
Рассчитывает целевую переменную `target` — суммарный GMV за следующие `horizon_days` после `anchor_date`.

### `process_all_folds(data, output_dir="data/v2/features", n_folds=4, batch_size=50_000)`
Запускает пакетную генерацию фичей и таргетов для всех 4-х валидационных фолдов и 1-го тестового фолда (`fold_end`), записывая Parquet-файлы на диск.

### `read_fold(output_dir="data/v2/features", fold_name="fold_03")`
Быстро загружает готовый фолд из сгенерированных `.parquet` батчей.

---

## 4. Быстрый пример использования в ноутбуке или скрипте

Вместо объемного кода генерации фичей теперь достаточно вызвать:

```python
import polars as pl
from src.data import read_fold, generate_cv_anchor_dates, process_all_folds

# 1. Загрузка исходных данных
data = pl.read_parquet("data/train.parquet")

# 2. Если нужно перегенерировать фичи заново (один раз)
# process_all_folds(data)

# 3. Чтение готового фолда для обучения
train_df = read_fold(fold_name="fold_03")
print(train_df.shape)  # (250000, 27)
```
