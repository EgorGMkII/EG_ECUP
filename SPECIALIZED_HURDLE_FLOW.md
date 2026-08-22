# Инженерное руководство: Specialized Hurdle Stack Pipeline (FLOW.md)

Данный документ содержит полное описание архитектуры, регламента выполнения и каталога типовых ошибок (Post-Mortem & Best Practices) для двухэтапного пайплайна **Specialized Hurdle Stack (RUN 1 + RUN 2)**.

---

## 1. Архитектурная схема пайплайна

```
                             [ data/train.parquet ] (30.6 млн строк)
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
  [ RUN 1: Purged Time-CV ]                                     [ RUN 2: Full Train & Submission ]
  • 23 якоря (2025-03-31 .. 2026-01-14)                         • Pooled Train (2.3M строк / 800k seq)
  • OOF-прогнозы по фолдам                                      • 1 финальный экземпляр каждого специалиста
  • Обучение метамоделей (Logit + Ridge)                        • Нативная генерация 250k test snapshot (2026-02-13)
        │                                                             │
        ▼                                                             ▼
  [ run1_meta_weights.json ] ──────────────────────────────────> [ Step C: Stacking с ALPHA=1.1 ]
  • w_react, w_churn, ridge_coef                                      │
                                                                      ▼
                                                       [ submission_specialized_hurdle_stack.csv ]
```

---

## 2. Пошаговый алгоритм выполнения

### Этап 1: RUN 1 — Получение и фиксация мета-весов (`scripts/run1_train_meta_weights.py`)
1. **Данные**: 23 канонических якорных даты (`2025-03-31 .. 2026-01-14`) с шагом 14 дней.
2. **Purged Time-CV**:
   * Разделение на временные фолды с изоляцией окон таргета (`A + 1d .. A + 30d`) и истории (`A - 364d .. A`).
   * Обучение специалистов (CatBoost, S1 Masked GRU, S2 Dense GRU, ETT).
   * Получение честных Out-Of-Fold (OOF) предсказаний на скрытых валидационных якорях.
3. **Обучение метамоделей**:
   * `Meta React`: Logistic Regression на спящих пользователях (`was_active == 0`).
   * `Meta Churn`: Logistic Regression на активных пользователях (`was_active == 1`).
   * `Meta Amount`: Ridge Regression на покупателях (`will_buy == 1`).
4. **Сохранение**: Веса и `ALPHA = 1.1` фиксируются в `artifacts/specialized_hurdle/run1_meta_weights.json`.

---

### Этап 2: RUN 2 — Финальное обучение и генерация Submission (`scripts/run2_train_and_submit.py`)

#### Step A: CatBoost Specialists
1. Сборка обучающей витрины на ВМ из `data/snapshots/`: 23 якоря x 100k = **2 300 000 строк**.
2. Генерация тестового снимка на лету для даты `2026-02-13` через `build_snapshot` для всех **250 000 пользователей** (время: ~15-20 сек).
3. Обучение 3 специалистов:
   * `CB_REACT_FINAL` (на `was_active == 0`);
   * `CB_CHURN_FINAL` (на `was_active == 1`);
   * `CB_AMOUNT_FINAL` (на `target > 0`).
4. Инференс на `X_test_cb` (250 000 строк) -> `cb_react_logits_test`, `cb_churn_logits_test`, `cb_amount_z_test`.
5. Освобождение памяти: `del df_cb_train, X_train_cb, df_cb_test, X_test_cb; gc.collect()`.

#### Step B: Neural Specialists (S1, S2, ETT)
1. Выбор 8 репрезентативных сезонных якорей годового цикла (800 000 обучающих сэмплов).
2. Выделение Memmap буферов на NVMe SSD (`np.float16` для тензоров `c` и `t`, `np.int16` для `ranks`, `bool` для `mask` и `empty`).
3. Извлечение последовательностей логов:
   * Обучающие: 8 якорей x 100k пользователей = 800 000 строк;
   * Тестовые: якорь `2026-02-13` x 250 000 пользователей = 250 000 строк.
4. Освобождение `df_raw`: `del df_raw; gc.collect()`.
5. Обучение моделей на GPU Tesla V100:
   * `S1 Masked GRU` (3000 шагов, batch 512, lr 5e-4);
   * `S2 Dense GRU` (3000 шагов, batch 512, lr 5e-4);
   * `Event-Time Transformer (180 токенов, tau=30d)` (4500 шагов, batch 512, lr 3e-4).
6. Параллельный тестовый инференс на 250k пользователях (`test_loader`, batch 1024).
7. Очистка буферов Memmap с диска.

#### Step C: Ансамблирование и сборка сабмита
1. Загрузка мета-весов из `run1_meta_weights.json`.
2. Вычисление `p_react = sigmoid(X_r * w_react)` и `p_churn = sigmoid(X_c * w_churn)`.
3. Определение `p_buy = np.where(was_act_test == 0, p_react, 1.0 - p_churn)`.
4. Вычисление `conditional_z = max(0, X_amt * ridge_coef + ridge_intercept)`.
5. Калибровка Hurdle: `z_pred = (p_buy ^ 1.1) * conditional_z`.
6. Перевод в рубли: `gmv = exp(z_pred) - 1`.
7. Сохранение в точном порядке `sample_submit.csv` -> `submission_specialized_hurdle_stack.csv`.

---

## 3. Каталог ошибок и проверенные решения (Lessons Learned & Anti-Patterns)

### 1. Передача тяжелых датасетов в облако (ConnectionResetError / Timeout)
* **Проблема**: Загрузка локально сгенерированных директорий с сотнями таблиц признаков (`artifacts/specialized_hurdle/feature_store/`) занимала десятки минут и падала по таймауту сети.
* **Решение**: Запрещено загружать готовые витрины признаков на ВМ. Передаются только исходные логи (`data/train.parquet`, `data/snapshots/`), а сборка витрин (`build_causal_feature_store` и `build_snapshot`) выполняется на лету на самой ВМ за 20 секунд. Пакет весит 1.1 МБ и отправляется за 1.6 секунды.

### 2. Несовпадение размерностей теста и трейна (IndexError: 100k vs 250k)
* **Проблема**: Исторические снапшоты в `data/snapshots/` содержали 100 000 отобранных пользователей. Если тестовый инференс для `2026-02-13` брал фолбэк на 100k, размерность массивов CatBoost и нейросетей расходилась со списком из 250 000 пользователей.
* **Решение**: Тестовая витрина на дату `2026-02-13` собирается вызовом `build_snapshot(df_raw, all_users, date(2026, 2, 13), is_test=True)` прямо на 250k пользователях.

### 3. Батчинг из Memmap и NumPy 2.x Type Casting
* **Проблема**: Конструкция `torch.from_numpy(np.array(memmap_slice, copy=False))` в NumPy 2.x выбрасывала `ValueError: Unable to avoid copy while creating an array as requested`.
* **Решение**: В методе `__getitem__` использовать явное приведение:
  ```python
  torch.from_numpy(np.asarray(self.content[idx], dtype=np.float32))
  ```

### 4. Выбор типа точности (float16 vs bfloat16 на GPU V100)
* **Проблема**: Попытка использовать `bfloat16` на GPU архитектуры Volta (Tesla V100, Compute Capability 7.0) приводит к программной эмуляции и замедлению в 3-4 раза.
* **Решение**: На Tesla V100 Tensor Cores аппаратно ускоряют только `float16` (FP16). `np.memmap` нативно работает с `np.float16`, что экономит 50% оперативной памяти и дискового пространства без потери точности.

### 5. Структура обучающей выборки нейросетей
* **Проблема**: Заполнение массива нулями для пользователей, отсутствующих в 100k снапшотах.
* **Решение**: Обучающие буферы нейросетей формируются строго по списку 100k обучающих пользователей: 8 якорей x 100k = 800 000 плотных строк без паразитных нулей. Тестовый буфер формируется строго на 250k пользователях.

---

## 4. Чек-лист перед запуском в DataSphere

1. [x] Проверить синтаксис: `python -m py_compile scripts/run2_train_and_submit.py src/*.py`.
2. [x] Убедиться, что в `datasphere.*.yaml` в `local-paths` включены:
   * `src/`
   * `scripts/`
   * `configs/`
   * `data/train.parquet`
   * `data/snapshots/`
   * `sample_submit.csv`
   * `artifacts/specialized_hurdle/run1_meta_weights.json`
3. [x] Убедиться, что тяжелые промежуточные папки (`artifacts/.../feature_store`) исключены из `local-paths`.
4. [x] Запуск через DataSphere Runner:
   ```bash
   python scripts/datasphere_runner.py -c datasphere.run2_final_submission.yaml
   ```
