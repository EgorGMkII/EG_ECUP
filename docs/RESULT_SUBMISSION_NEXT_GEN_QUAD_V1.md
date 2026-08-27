# RESULT: Next-Generation Quad Stack & Grand Champion Blend Submissions (v1)

- **Job ID**: `bt1n2e9pa3592plu15bk`
- **Instance**: GPU `g1.1`
- **Cohort**: Full 250,000 users
- **Date**: 2026-08-27

---

## 1. Сгенерированные сабмиты

### 1. `submission_direct_next_gen_quad_stack_v1.csv`
- **Состав**:
  - `direct_frequency_specialist` (40%): CatBoost для частых покупателей + LightGBM для спящих на прямом таргете.
  - `direct_delta_regressor` (28%): CatBoost на дельтах $\Delta Z = Z_{\text{true}} - \ln(1 + \text{gmv\_sum\_30d})$.
  - `two_tower_direct` (17%): Двухбашенная нейросеть (Activity Tower + Monetary Tower) на сырых событиях.
  - `catboost_direct` (15%): CatBoost на 226 признаках (194 sparse + 32 CoLES).
- **Метрики распределения**:
  - Mean GMV: **32.94** руб.
  - Median GMV: **6.35** руб.
  - Max GMV: **2727.84** руб.
- **SHA-256**: `832bce83b7e0073b...`

### 2. `submission_grand_next_gen_champion_blend_v1.csv` (Рекомендован к отправке!)
- **Состав**:
  - `0.60` $\times$ `submission_meta_blend_champions_v1.csv` (Текущий рекорд LB: **1.655904**)
  - `0.40` $\times$ `submission_direct_next_gen_quad_stack_v1.csv` (Next-Gen Quad Stack)
- **Метрики распределения**:
  - Mean GMV: **35.66** руб.
  - Median GMV: **6.68** руб.
  - Max GMV: **2387.86** руб.
- **SHA-256**: `84cc2ff0245666c3...`

---

## 2. Результаты валидации и проверок

- Формат колонок: `user_id`, `predict`.
- Количество строк: ровно 250 000.
- Пропуски (NaN/Null/Inf): **0**.
- Отрицательные значения: **0**.
- Полное совпадение порядка `user_id` с `sample_submit.csv`: **100%**.
