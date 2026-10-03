# Отчет по каноническому эксперименту на 250k пользователях (Canonical Specialized Hurdle 250k)

## 1. Концепция и цель эксперимента
Воспроизведение 1-в-1 канонической методики подбора метавесов и раздельного обучения из эксперимента со 100k пользователями (Public LB: 1.664077), но с обучением финальных специалистов на полном пуле данных (все 250k пользователей, 23 якоря, 2 300 000 строк).

---

## 2. Параметры метавесов (RUN 1 на 17 якорях)
Подбор метавесов проводился на валидационном якоре `2025-12-15` (модели обучались на 17 якорях до `2025-11-24`):

1. **React Stack (Simplex LogLoss)**:
   - CatBoost: `0.0763`
   - S1 Masked GRU: `0.2484`
   - S2 Dense GRU: `0.0000`
   - Event-Time Transformer: `0.6753`
   - *Сумма весов*: `1.0000`

2. **Churn Stack (Simplex LogLoss)**:
   - CatBoost: `0.4691`
   - S1 Masked GRU: `0.0000`
   - S2 Dense GRU: `0.0000`
   - Event-Time Transformer: `0.5309`
   - *Сумма весов*: `1.0000`

3. **Amount Positive Ridge (Regression)**:
   - CatBoost: `0.1606`
   - S1 Masked GRU: `0.0523`
   - S2 Dense GRU: `0.2745`
   - Event-Time Transformer: `0.5147`
   - Intercept: `-0.0468`
   - *Сумма коэффициентов*: `1.0021` (масштаб логарифмического чека сохранен без искусственного занижения)

4. **Параметр нелинейности Hurdle**:
   - `alpha = 1.1`
   - `z_pred = (p_buy ^ 1.1) * cond_z`
   - `GMV = exp(z_pred) - 1`

---

## 3. Финальное обучение специалистов (RUN 2)
- **Обучающая выборка**: все 23 якоря (`2025-03-31` .. `2026-01-14`), 2 300 000 строк (384 признака).
- **Модели**:
  - `CatBoost React` (LogLoss, depth=6, GPU)
  - `CatBoost Churn` (LogLoss, depth=6, GPU)
  - `CatBoost Amount` (RMSE on ln(1+GMV), depth=6, GPU)
  - `S1 Masked GRU` (hidden=64, mask threshold=30d, GPU)
  - `S2 Dense GRU` (hidden=64, delta features, GPU)
  - `Event-Time Transformer` (180 tokens, d_model=64, 4 heads, tau=30d, GPU)
- **Инференс**: тестовый срез `2026-02-13` для всех 250 000 пользователей.

---

## 4. Статистика и сравнение сабмитов

| Метрика / Сабмит | Canonical 250k (НОВЫЙ) | Exact 100k weights on 250k (Рекорд LB) | Joint SLSQP 250k v2 |
| :--- | :--- | :--- | :--- |
| **Файл** | `submission_specialized_hurdle_canonical_250k.csv` | `submission_specialized_hurdle_joint_250k_exact.csv` | `submission_specialized_hurdle_joint_250k_v2.csv` |
| **Public LB** | *(готов к отправке)* | **1.664077** | 1.667658 |
| **Mean GMV** | **31.26 RUB** | 47.09 RUB | 37.72 RUB |
| **Median GMV**| **5.22 RUB** | 8.87 RUB | 6.84 RUB |
| **Min GMV** | 0.12 RUB | 0.00 RUB | 0.00 RUB |
| **Max GMV** | 3044.20 RUB | 4126.50 RUB | 3582.10 RUB |
| **Pearson r с рекордом** | **0.9876** | 1.0000 | 0.9913 |
| **Spearman rho с рекордом**| **0.9957** | 1.0000 | 0.9957 |
| **Размерность** | 250,000 x 2 (`user_id`, `predict`) | 250,000 x 2 | 250,000 x 2 |
| **Пропуски (NaN)** | 0 | 0 | 0 |
| **Совпадение user_id** | 100% | 100% | 100% |
