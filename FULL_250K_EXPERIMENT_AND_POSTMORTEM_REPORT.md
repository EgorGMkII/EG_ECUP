# Официальный эталонный отчет: Рекордный сабмит и Post-Mortem экспериментов

## 1. Официальная сводная таблица всех сабмитов на Leaderboard

| Эксперимент / Сабмит | Выборка обучения базовых моделей | Метаоптимизация | Mean GMV | Median GMV | Public LB RMSLE | Статус |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🏆 [**`submission_specialized_hurdle_joint_rmsle.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_rmsle.csv) | **100k стратифицированная** (23 якоря) | **Joint SLSQP (End-to-End RMSLE)** | **43.21 RUB** | **7.65 RUB** | **1.6640779122** | 🥇 **АБСОЛЮТНЫЙ РЕКОРД СОРЕВНОВАНИЯ** |
| 🥈 [**`submission_specialized_hurdle_stack.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_stack.csv) | **100k стратифицированная** (23 якоря) | **Раздельный подбор (LogLoss + Positive Ridge)** | **38.03 RUB** | **6.45 RUB** | **1.6649359945** | **Top-2 LB (Baseline)** |
| 🥉 [**`submission_specialized_hurdle_joint_250k_exact.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_250k_exact.csv) | 250k ранние веса | Joint метавеса (ранний срез) | **47.09 RUB** | **6.45 RUB** | **1.6657399037** | Уступает рекорду |
| 4. [**`submission_specialized_hurdle_joint_250k_v2.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_250k_v2.csv) | 250k когорта (все 23 якоря) | Joint SLSQP (на Dec 15 без рег.) | **37.72 RUB** | **5.80 RUB** | **1.6676582091** | Деградация из-за шума 250k |
| 5. [**`submission_specialized_hurdle_canonical_250k.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_canonical_250k.csv) | 250k когорта (все 23 якоря) | Раздельный подбор (LogLoss + Ridge) | **31.26 RUB** | **5.22 RUB** | **1.6764263333** | Занижение чека из-за 250k шума |
| 6. [**`submission_specialized_hurdle_joint_250k_temporal_regularized.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_250k_temporal_regularized.csv) | 250k когорта (все 23 якоря) | Joint SLSQP (Dec+Jan) + жесткий L2 штраф | **25.45 RUB** | **4.11 RUB** | **1.6882920404** | Худший (перерегуляризация) |
| 7. `submission_specialized_hurdle_joint_250k_v1_buggy` | 250k когорта | Ошибочная нормализация Amount | 192.80 RUB | 0.04 RUB | **2.316567** | Сломанный масштаб |

---

## 2. Полное описание рекордного пайплайна `submission_specialized_hurdle_joint_rmsle.csv` (LB: 1.6640779122)

### А. Скрипты и файлы конфигурации:
1. **Исполняемый скрипт инференса и сборки**:
   [`scripts/build_joint_rmsle_submission.py`](file:///c:/Users/egorg/Documents/OZON_ECUP/scripts/build_joint_rmsle_submission.py)
2. **DataSphere Манифест**:
   [`datasphere.joint_submission.yaml`](file:///c:/Users/egorg/Documents/OZON_ECUP/datasphere.joint_submission.yaml)
3. **Скрипт проверки и аудита**:
   [`scripts/verify_joint_submission.py`](file:///c:/Users/egorg/Documents/OZON_ECUP/scripts/verify_joint_submission.py)
4. **Конфигурация оптимизированных метавесов**:
   [`artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json)
5. **Банк сырых предсказаний специалистов на 250k тесте**:
   [`test_specialists_raw_predictions_250k.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/test_specialists_raw_predictions_250k.parquet)
6. **Диагностическая таблица предсказаний (250k)**:
   [`submission_specialized_hurdle_joint_rmsle_diagnostics.parquet`](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_rmsle_diagnostics.parquet)

---

### Б. Архитектура и точные параметры моделей:

1. **CatBoost Specialists (Деревья решений)**:
   - Обучающая выборка: 23 якоря со стратификацией (`selected_users_100k.parquet`), 375 признаков.
   - `CB_REACT`: iterations=1500, lr=0.04, depth=6, loss=Logloss на `was_active == 0`.
   - `CB_CHURN`: iterations=1500, lr=0.04, depth=6, loss=Logloss на `was_active == 1`.
   - `CB_AMOUNT`: iterations=1500, lr=0.04, depth=6, loss=RMSE на `target > 0` по `ln(1 + GMV)`.

2. **Нейросетевые последовательностные специалисты (Sequential Neural Models)**:
   - `S1 Masked GRU`: 2 слоя GRU, d_model=128, dropout=0.1, маска неактивности (порог 30 дней), lr=5e-4, AdamW.
   - `S2 Dense GRU`: 2 слоя GRU, d_model=128, непрерывный поток плотных дельт, lr=5e-4, AdamW.
   - `Event-Time Transformer`: 2 слоя Transformer Encoder, 4 heads, d_model=128, dim_feedforward=512, gelu, max_events=180, временное затухание `tau = 30d`, сезонная гармоника (sin/cos DOY), lr=3e-4, AdamW.
   - Функция потерь нейросетей: `Loss = MSE(direct_z) + 0.5 * MSE(cond_z) + 0.25 * (BCE_react + BCE_churn)`.

---

### В. Метаоптимизация (Joint RMSLE SLSQP):
- **Схема валидации**: 5-Fold Cross-Validation и 70k/30k Out-of-Sample Split на OOF предсказаниях 100k пользователей (`run1_meta_predictions.parquet`, якорь `2025-12-15`).
- **Алгоритм**: Scipy `minimize(method="SLSQP")` с целевой функцией `RMSLE(y_true, y_pred)`.
- **Точные зафиксированные метавеса (13 параметров)**:
  * **React Stack (`was_active == 0`)**:
    - CatBoost: `0.02571986`
    - S1 Masked GRU: `0.36589676`
    - S2 Dense GRU: `0.00000000`
    - Event-Time Transformer: `0.60838337` (основной драйвер реактиваций)
  * **Churn Stack (`was_active == 1`)**:
    - CatBoost: `0.27705640`
    - S1 Masked GRU: `0.22700033`
    - S2 Dense GRU: `0.06279934`
    - Event-Time Transformer: `0.43314393`
  * **Amount Ridge (Коэффициенты чека)**:
    - CatBoost: `0.12217097`
    - S1 Masked GRU: `0.00000000`
    - S2 Dense GRU: `0.35635864` (ключевой драйвер чека)
    - Event-Time Transformer: `0.54547271` (главный предиктор чека)
    - **Ridge Intercept**: **`+0.02079581`** (положительный сдвиг)
  * **Hurdle Экспонента**:
    - `ALPHA = 1.1` (строго зафиксирован)
  * **Формула сборки**:
    - `p_buy = np.where(was_active == 0, expit(X_r @ w_r), 1.0 - expit(X_c @ w_c))`
    - `cond_z = np.maximum(0.0, X_a @ w_a + b_a)`
    - `z_pred = np.clip((p_buy ** 1.1) * cond_z, 0.0, None)`
    - `GMV = np.expm1(z_pred)`

---

## 3. Детальный разбор: Почему 250k уступает 100k стратификации

1. **Размытие покупательского сигнала (Data Dilution)**:
   - В 100k стратифицированной выборке соотношение активных покупателей к спящим было сбалансировано.
   - При добавлении 150 000 разреженных пользователей из общего пула градиенты деревьев CatBoost и градиенты нейросетей сместили базовые распределения чеков и вероятностей вниз.
   - Из-за этого средний GMV упал с 43.2 RUB до 31-37 RUB, вызвав недопрогноз на реальных покупках.
2. **Ошибка раздельного обучения (LogLoss отдельно, Ridge отдельно)**:
   - Раздельный Ridge подбирал отрицательный интерсепт (`-0.0468`), что в связке с формулой `exp(z) - 1` срезало крупные чеки.
   - Joint SLSQP напрямую согласовывал произведение `(p_buy ** 1.1) * cond_z` с таргетом, выставив положительный интерсепт `+0.0208`.
3. **Провал межсезонной регуляризации**:
   - Наложение штрафа L2 при объединении декабря и января искусственно сжало веса и уронило LB до 1.6882.

---

## 4. Итоговая фиксация

* **Главный сабмит соревнования**: [**`submission_specialized_hurdle_joint_rmsle.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/submission_specialized_hurdle_joint_rmsle.csv) (**Public LB: 1.6640779122**).
* **Скрипт воспроизведения**: [`scripts/build_joint_rmsle_submission.py`](file:///c:/Users/egorg/Documents/OZON_ECUP/scripts/build_joint_rmsle_submission.py).
