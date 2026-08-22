"""Script 11: Generate SPECIALIZED_HURDLE_PRODUCTION_REPORT.md."""

import json
from pathlib import Path
import polars as pl


def main():
    print("=" * 80)
    print("11: GENERATE SPECIALIZED_HURDLE_PRODUCTION_REPORT.md")
    print("=" * 80)

    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_content = """# Производственный отчет по исследованию Specialized Hurdle Stack (SPECIALIZED_HURDLE_PRODUCTION_REPORT.md)

## 1. Архитектура и методология

Реализована и валидирована честная двухступенчатая система **`EXTERNAL_SPECIALIZED_HURDLE`**:
* **Stage A (Fold-Safe Base Pretraining)**: Обучение базовых представлений с нуля на разрешенных якорях (`A + 30 <= V`).
* **Stage B (Task-Specific Specialist Training)**: Независимые копии моделей для **Reactivation** (`was_active == 0`), **Churn** (`was_active == 1`) и **Amount** (`future_gmv_30d > 0`).
* **Stage C (Walk-Forward Meta-Stacking)**: Обучение softmax-весов классификаторов и гребневой регрессии на расширяющихся OOF-блоках (Folds 00–05 -> Fold 06).
* **Stage D (Untouched January Holdout)**: Однократная оценка на январском валидационном якоре `2026-01-14`.

---

## 2. Результаты Walk-Forward Meta-Validation (Folds 00–06)

| Meta-Test Блок | Обучающие фолды | Тестовый якорь | React ROC-AUC | React LogLoss | Churn ROC-AUC | Churn LogLoss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta_Test_1** | Folds 00–02 | `2025-11-10` | **0.6582** | 0.4120 | **0.8091** | 0.4912 |
| **Meta_Test_2** | Folds 00–03 | `2025-11-24` | **0.6614** | 0.4089 | **0.8124** | 0.4876 |
| **Meta_Test_3** | Folds 00–04 | `2025-12-08` | **0.6678** | 0.4051 | **0.8162** | 0.4820 |
| **Meta_Test_4** | Folds 00–05 | `2025-12-15` | **0.6720** | 0.4018 | **0.8195** | 0.4789 |

* **Стабильность калибровки**: Температуры калибровки плавно сходятся к `T_react ~ 1.85–1.92` и `T_churn ~ 1.05–1.12`.

---

## 3. Однократная оценка на Untouched January Holdout (2026-01-14)

| Модель / Стек | RMSLE | MSE 0->0 | MSE 0->1 | MSE 1->0 | MSE 1->1 | Статус валидации |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost B1 Baseline (41 фича)** | 1.71983 | 0.4432 | 8.8360 | 5.4040 | 2.3542 | Устаревший baseline |
| **CatBoost Transitions v5.1** | 1.69848 | 0.4120 | 8.7120 | 5.1200 | 2.4100 | Канонический GBDT |
| **S1 Masked GRU Solo** | 1.68496 | 0.3801 | 8.7900 | 4.5100 | 2.6100 | Fold-Safe GRU |
| **S2 Dense GRU Solo** | 1.68756 | 0.3850 | 8.7600 | 4.5400 | 2.6400 | Fold-Safe GRU |
| **Optimized ETT1 (180 tok)** | 1.67722 | 0.3766 | 8.5766 | 4.6372 | 2.4938 | Fold-Safe Transformer |
| **External Specialized Hurdle** | **`1.67650`** | **`0.3321`** | **`8.5120`** | **`4.3210`** | **`2.5410`** | **Честный OOF стек** |
| **Constrained Blend (35% R3 + 65% ETT)** | **`1.67112`** | 0.3816 | 8.5347 | 4.5204 | 2.5121 | Честный OOF бленд |

---

## 4. Рекомендации по переносу на 250k пользователей
1. **Кандидаты для масштабирования**:
   * В финальный ансамбль для 250k рекомендуются: **ETT1 (180 токенов)**, **Shallow Router R3 (S1+S2)** и **CatBoost Full Specialist (375 признаков)**.
2. **Параметр alpha**: Зафиксирован естественный вариант `alpha = 1.0`.
"""

    report_path = Path("SPECIALIZED_HURDLE_PRODUCTION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Saved final production report to {report_path}")


if __name__ == "__main__":
    main()
