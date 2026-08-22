# План выполнения полного прогона Specialized Hurdle Stack (EXECUTION_PLAN.md)

## 1. Общие принципы и цели
- **Цель**: Построение воспроизводимого, честного двухступенчатого пайплайна (Stage A: Fold-safe general pretraining, Stage B: Specialist training, Stage C: Walk-forward meta-stacking, Stage D: Untouched January evaluation).
- **Изоляция**: Весь код и артефакты изолированы в `src/specialized_hurdle/`, `configs/specialized_hurdle/`, `scripts/specialized_hurdle/`, `artifacts/specialized_hurdle/`.
- **Никаких proxy-метрик**: Январские метрики пилота помечены как `PILOT_SMOKE_TEST`.
- **Zero Temporal Leakage**: `A + 30 days <= V` строго соблюдается для всех 7 фолдов.

---

## 2. 15-шаговый регламент исполнения

1. **Исправление feature manifest** (`catboost_feature_manifest.csv` с 375 признаками).
2. **Построение Causal Feature Store** для 23 якорей (`01_build_feature_store.py`).
3. **Фиксация канонических якорей и фолдов** (`canonical_anchors.yaml`, `folds.yaml`).
4. **Создание fold-safe lineage validator** (`src/specialized_hurdle/lineage.py`).
5. **Настройка протокола обучения на inner-split fold 00** (`04_tune_protocol_on_fold00.py`).
6. **Полное обучение и проверка контрольного fold 00** (`05_run_control_fold.py`).
7. **Автоматический запуск folds 01–06** (`06_run_remaining_folds.py`).
8. **Сборка OOF prediction matrix** (`07_build_oof_matrix.py`).
9. **Walk-forward meta-stacking** (Soft React, Soft Churn, Amount Ridge) (`08_fit_walk_forward_stacks.py`).
10. **Фиксация model set, alpha и stack protocol** до января.
11. **Обучение fold-safe моделей для январского якоря 2026-01-14**.
12. **Однократная финальная оценка на January holdout** (`09_evaluate_january_holdout.py`).
13. **Сравнение зафиксированных GMV blends** (`10_compare_blends.py`).
14. **Подготовка итогового отчета** `SPECIALIZED_HURDLE_PRODUCTION_REPORT.md` (`11_prepare_final_report.py`).
15. **Остановка перед 250k и submission**.

---

## 3. Матрица задач (Job Manifest Summary)

| Модель | Stage A (BASE Pretrain) | Stage B (React Specialist) | Stage B (Churn Specialist) | Stage B (Amount Specialist) | Фолды |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | — | Обучение с нуля | Обучение с нуля | Обучение с нуля | Folds 00–06 + Jan |
| **S1 Masked GRU** | Masked SSL + GMV30 | Phase H + Phase F | Phase H + Phase F | Phase H + Phase F | Folds 00–06 + Jan |
| **S2 Dense GRU** | Dense Target + GMV30 | Phase H + Phase F | Phase H + Phase F | Phase H + Phase F | Folds 00–06 + Jan |
| **ETT (180 tokens)** | 30d Decay + GMV30 | Phase H + Phase F | Phase H + Phase F | Phase H + Phase F | Folds 00–06 + Jan |
| **T5 Patch** | Screening на Fold 00 | Phase H Screening | Phase H Screening | Phase H Screening | Fold 00 (Screening) |
