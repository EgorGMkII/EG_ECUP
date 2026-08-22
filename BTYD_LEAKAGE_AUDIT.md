# Отчет об аудите утечки и честном повторе BTYD-экспериментов (BTYD_LEAKAGE_AUDIT.md)

---

## 1. Главный вывод аудита: TARGET LEAKAGE FOUND AND FIXED

В ходе проверки предыдущего отчета [`BTYD_RESEARCH_REPORT.md`](file:///c:/Users/egorg/Documents/OZON_ECUP/BTYD_RESEARCH_REPORT.md) была обнаружена и устранена грубая утечка целевой переменной (**Target Leakage**).

### 1.1. Точная первопричина и строки кода
* **Файл ошибки**: [`scripts/run_btyd_research_experiments.py`](file:///c:/Users/egorg/Documents/OZON_ECUP/scripts/run_btyd_research_experiments.py), строка **266**.
* **Фрагмент кода, вызвавший сбой**:
  ```python
  base_feat_cols = [c for c in val_snap.columns if c not in ["user_id", "anchor_date", "target", "current_state", "global_dau", "vs_global_orders_30d", "vs_global_orders_7d", "vs_global_gmv_30d", "vs_global_gmv_7d"]]
  ```
  Вместо вызова канонической функции [`get_feature_columns()`](file:///c:/Users/egorg/Documents/OZON_ECUP/src/hurdle.py#L24-L26) был использован неполный список исключений.

### 1.2. Механизм утечки и почему AUC стал равен 1.0000
В паркет-файле снапшота присутствовала колонка **`will_buy_30d`** — бинарный истинный таргет `(target > 0).cast(int)`. 
* Линейная корреляция колонки `will_buy_30d` с таргетом классификатора: **`1.000000`**.
* Классификатор первого этапа получил на вход точный ответ о факте совершения покупок, что привело к фиктивным `React AUC = 1.0000`, `Churn AUC = 1.0000`, `Brier = 4.0e-7` и искусственному занижению `RMSLE` до `0.82672` (остаточная ошибка регрессора на покупателях).

### 1.3. Статус предыдущих результатов
* Эксперименты T3, T4, T5 из `BTYD_RESEARCH_REPORT.md` объявлены **НЕДЕЙСТВИТЕЛЬНЫМИ (INVALID_DUE_TO_TARGET_LEAKAGE)**.
* Предыдущий вывод «BTYD ухудшает CatBoost» аннулирован как полученный на протекающем бейзлайне.

---

## 2. Проверка гипотез аудита (Разделы 2.1–2.6)

1. **2.1. Пересечение train и validation**:
   * Набор `(user_id, anchor_date)` валидационного якоря `2026-01-14` полностью изолирован от 8 обучающих якорей (0 общих строк, `intersection == 0`).
2. **2.2. Target-derived columns**:
   * В паркет-файле найдена и полностью исключена колонка `will_buy_30d`. Все остальные 368 базовых признаков строго ретроспективны.
3. **2.3. Personal Propensity Leakage**:
   * Признаки `hist_target_*` используют строго якоря `j`, для которых `target_end_date(j) <= a`. Валидационный таргет не используется.
4. **2.4. Cadence & Feature Joins**:
   * Все события в `extract_cadence_features_for_anchor` и `extract_full_history_rfm_for_anchor` ограничены условием `event_date <= anchor_date`.
5. **2.5. In-sample evaluation**:
   * Модели обучались строго на обучающих якорях панели, валидация проводилась строго out-of-time на `2026-01-14`.

---

## 3. Восстановление канонического паритета (Baseline C4 Parity)

После восстановления вызова `get_feature_columns()` и сборки честной матрицы из 421 признака (368 Base + 36 Cadence + 8 Propensity + 9 Last-Year) получен результат бейзлайна **B0 (Честный C4)**:

| Метрика | Значение в эксперименте B0 | Канонический ориентир C4 | Разница (Delta) | Статус проверки |
| :--- | :---: | :---: | :---: | :---: |
| **Validation RMSLE** | **1.684313** | **1.68431** | **+0.000003** | **100% ПОЛНОЕ СОВПАДЕНИЕ** |
| **Validation MSE log** | **2.836911** | 2.83690 | +0.00001 | Полное совпадение |
| **React ROC-AUC** | **0.76086** | **0.7609** | -0.00004 | Полное совпадение |
| **Churn ROC-AUC** | **0.80639** | **0.8064** | -0.00001 | Полное совпадение |
| **Overall Brier** | **0.15647** | **0.1565** | -0.00003 | Полное совпадение |
| **Число признаков** | **421** | **421** | 0 | Полное совпадение |

Паритет канонического бейзлайна C4 полностью и безупречно восстановлен.

---

## 4. Математическая семантика BTYD и Standalone-оценка

### 4.1. Семантика вероятности покупки
Формула `p_buy_proxy_30d = 1 - exp(-expected_purchases_30d)` является пуассоновским приближением к вероятности совершить хотя бы одну транзакцию в окне $T=30$ дней. В документации и коде признак зафиксирован как **`p_buy_proxy_30d`**.

### 4.2. Честная State-Specific оценка BTYD (Раздел 5)
Файл: [`artifacts/btyd_audit/btyd_state_metrics.json`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/btyd_state_metrics.json).

* **Вся выборка (N = 100 000)**:
  * Overall ROC-AUC: **0.83995**
  * Overall PR-AUC: **0.86300**
  * Overall Brier: **0.16725**
  * Средняя предсказанная вероятность: **0.5496** vs Фактическая доля: **0.5398**.
* **Подгруппа реактивации (`past_30d_gmv == 0`, N = 43 798)**:
  * Фактический Reactivation Rate: **27.96%**
  * Standalone React ROC-AUC: **0.7423**
  * Standalone React PR-AUC: **0.5355**
  * Standalone React Brier: **0.1802**
  * F1-score (порог 0.5): **0.5062** (Precision: 54.1%, Recall: 47.5%)
* **Подгруппа оттока (`past_30d_gmv > 0`, N = 56 202)**:
  * Фактический Churn Rate: **25.73%**
  * Standalone Churn ROC-AUC: **0.7849**
  * Standalone Churn PR-AUC: **0.5486**
  * Standalone Churn Brier: **0.1572**
  * F1-score (порог 0.5): **0.4260** (Precision: 60.6%, Recall: 32.8%)

---

## 5. Результаты минимального честного набора сравнений (B0–B3)

Эксперименты выполнены на идентичной обучающей панели (8 якорей), валидационном якоре `2026-01-14`, одинаковом random seed (42) и одинаковых гиперпараметрах CatBoost (600 итераций, depth=6, lr=0.05).

Таблица из [`artifacts/btyd_audit/experiment_registry.csv`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/experiment_registry.csv):

| Эксперимент | Cls Feats | Reg Feats | Валидационный RMSLE | MSE log | Delta RMSLE | React AUC | Churn AUC | Overall Brier | Bootstrap P(Better) | MSE 0→>0 (React) | MSE >0→>0 (Active) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0_Honest_C4** | 421 | 421 | **1.68431** | 2.83691 | 0.00000 | 0.76086 | 0.80639 | 0.15647 | База | 5.8112 | 1.9177 |
| **B1_BTYD_ProbCount_ClassifierOnly** | **424** | **421** | **1.68230** | **2.83015** | **-0.00201** | **0.76459** | **0.80792** | **0.15625** | **100.0%** | **5.6032** | **1.8817** |
| **B2_BTYD_Monetary_RegressorOnly** | 421 | 422 | **1.68477** | 2.83844 | +0.00045 | 0.76086 | 0.80639 | 0.15647 | 0.7% | 5.8110 | 1.9155 |
| **B3_BTYD_Target_Combination** | 424 | 422 | **1.68254** | 2.83095 | **-0.00177** | **0.76459** | **0.80792** | **0.15625** | **100.0%** | **5.6029** | **1.8787** |

---

## 6. Анализ инкрементального сигнала BTYD

1. **Арифметика и валидация инвариантов**:
   * `abs(1.6823039^2 - 2.8301466) = 0.0000000` (точность до $10^{-7}$).
   * `sum(group_N) = 100000` (100% строк учтены).
2. **Инкрементальный вклад BTYD в классификацию (B1)**:
   * Добавление только трех BTYD-признаков (`btyd_p_buy_30d`, `btyd_expected_purchases_30d`, `btyd_p_alive`) **исключительно в классификатор** дает устойчивое улучшение **`-0.00201` RMSLE** ($1.68431 \to 1.68230$).
   * Качество дискриминации реактивации вырастает: `React AUC` увеличивается с **0.7609 до 0.7646 (+0.0037)**.
   * Качество оттока вырастает: `Churn AUC` увеличивается с **0.8064 до 0.8079 (+0.0015)**.
   * Ошибка на реактивирующихся пользователях (`MSE 0->>0`) снижается с **5.8112 до 5.6032 (-3.6% SSE)**.
   * Бутстрап-тест: вероятность того, что B1 превосходит B0, равна **100.0%** (`Bootstrap_P_Better = 1.000`).
3. **Неэффективность BTYD Monetary в регрессоре (B2)**:
   * Добавление `btyd_expected_monetary_value` в регрессор дает микро-ухудшение на **+0.00045 RMSLE**, так как прямое предсказание чека по эмпирическим окнам 30/60/90d в CatBoost точнее сглаженного чека Gamma-Gamma.
4. **Корреляция ошибок**:
   * Линейная корреляция вероятностей CatBoost и BTYD: **0.9411**.
   * BTYD исправляет грубые ошибки CatBoost у **3.94%** пользователей, привнося гладкий параметрический априорный сигнал жизненного цикла.

---

## 7. Окончательное решение по BTYD

### РЕШЕНИЕ: BTYD ПРОШЕЛ АУДИТ И ПРИНЯТ В КЛАССИФИКАТОРЫ (КОНФИГУРАЦИЯ B1)

1. Улучшение `overall RMSLE` превысило порог: **$|\Delta \text{RMSLE}| = 0.00201 \ge 0.0020$**.
2. Улучшение `React AUC` (+0.0037) и `Churn AUC` (+0.0015) привело к реальному сокращению ошибки переходов $0 \to >0$ и $>0 \to >0$.
3. **Итоговое правило интеграции**:
   * Три признака BTYD (`btyd_p_buy_30d`, `btyd_expected_purchases_30d`, `btyd_p_alive`) **включаются в каноническую витрину классификаторов первого этапа**.
   * Признаки Gamma-Gamma monetary исключаются из conditional regressor.
4. Сохранены все паркет-файлы предсказаний:
   * [`artifacts/btyd_audit/predictions_B0.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/predictions_B0.parquet)
   * [`artifacts/btyd_audit/predictions_B1.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/predictions_B1.parquet)
   * [`artifacts/btyd_audit/predictions_B2.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/predictions_B2.parquet)
   * [`artifacts/btyd_audit/predictions_B3.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/btyd_audit/predictions_B3.parquet)
