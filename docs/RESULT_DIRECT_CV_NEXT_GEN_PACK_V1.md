# RESULT: Direct 4-Fold CV Next-Generation Experiment Pack (v1)

- **Job ID**: `bt1lff0c5vcj03iak988`
- **Instance**: GPU `g1.1`
- **Execution Time**: ~43 minutes
- **Cohort**: 250,000 users
- **Protocol**: 4-Fold Temporal CV (`FOUR_FOLD_250K_V1`)

---

## 1. Исследованные архитектуры нового поколения

1. **`catboost_direct`**: Референсный CatBoost на полном наборе из 226 признаков (194 воронкочно-каталожных + 32 CoLES dense).
2. **`direct_frequency_specialist`**:
   - Автоматическое разделение выборки на когорту постоянных покупателей ($\ge 2$ заказов за 90 дней) и спящих/редких пользователей.
   - Независимое обучение CatBoost для регулярных и LightGBM для спящих на прямом $Z = \ln(1+Y)$ таргете.
3. **`direct_delta_regressor`**:
   - Обучение на дельтах/остатках относительно 30-дневного базового чека: $\Delta Z = Z_{\text{true}} - \ln(1 + \text{gmv\_sum\_30d})$.
   - Реконструкция предсказания: $Z_{\text{pred}} = \max(0, Z_{\text{base}} + \widehat{\Delta Z})$.
4. **`two_tower_direct`**:
   - Двухбашенная глубокая нейросеть на сырых ежедневных последовательностях: **Activity Tower** (GRU/Conv1D на активностях) + **Monetary Tower** (GRU/Conv1D на суммах GMV) + Multi-Task Fusion Head.

---

## 2. Результаты 4-Fold Cross-Validation

| Модель | Fold F1 (Окт) | Fold F2 (Ноя) | Fold F3 (Дек) | Fold F4 (Holdout Янв) | Mean F1–F3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Direct (226f)** | 1.702211 | 1.735698 | 1.750972 | 1.687594 | 1.729627 |
| **Direct Frequency Specialist** | 1.703055 | 1.738656 | **1.744726** 🔥 | 1.689435 | 1.728812 |
| **Direct Delta Regressor** | 1.702789 | 1.737043 | **1.747823** 💥 | 1.689826 | 1.729218 |
| **Two-Tower Sequential Net** | 1.719900 | 1.760287 | 1.771536 | 1.713979 | 1.750574 |
| **Quad Next-Gen Synergy Blend** | **1.699841** | **1.734008** | **1.744072** | **1.686415** | **1.725974** |

---

## 3. Матрица корреляций предсказаний на Holdout Fold F4

| Модель | CatBoost | Freq Specialist | Delta Regressor | Two-Tower Net |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | 1.0000 | 0.9966 | 0.9969 | **0.9749** |
| **Freq Specialist** | 0.9966 | 1.0000 | 0.9950 | **0.9735** |
| **Delta Regressor** | 0.9969 | 0.9950 | 1.0000 | **0.9730** |
| **Two-Tower Net** | 0.9749 | 0.9735 | 0.9730 | 1.0000 |

---

## 4. Оптимальные веса ансамбля (строго по F1–F3 без утечки в F4)

- `direct_frequency_specialist`: **0.3999** (40.0%)
- `direct_delta_regressor`: **0.2815** (28.2%)
- `two_tower_direct`: **0.1696** (17.0%)
- `catboost_direct`: **0.1490** (14.9%)

### Прирост на Holdout Fold F4:
- CatBoost Baseline: `1.687594`
- Quad Next-Gen Blend: `1.686582` (**`-0.001011`**)
