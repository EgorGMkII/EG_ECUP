# Submission provenance

Итог команды, сообщённый автором: RMSLE **1.6558615631**, **152/316**.
Источник: https://ods.ai/tracks/e-cup-2026-competitions/competitions/e-cup-2026-search/leaderboard
Страница не была доступна через инструмент чтения; результат атрибутирован автору.

| Recipe | Свидетельства | Состав |
|---|---|---|
| `build_direct_cb_ett_submission.py` | `artifacts/direct_CB_ETT/manifest.json`, Public 1.6572245219 из диалога | CB 0.755099 + direct ETT 0.244901 |
| `build_direct_coles209_triple_stack_submission.py` | `artifacts/direct_coles209_triple_stack_submission_v1/manifest.json`, PRE-RUN e3fcb338… | 209f; CB/LGB/multi-task ETT 0.38/0.32/0.30 |
| `build_direct_multi_anchor_750k_triple_stack_submission.py` | `artifacts/direct_multi_anchor_750k_triple_stack_v1/manifest.json`, PRE-RUN bb4f68fc… | 3 train anchors, 750k user-anchor rows; CB/LGB/ETT |
| `build_direct_226f_triple_stack_submission.py` | `artifacts/direct_226f_triple_stack_submission_v1/manifest.json` | Sparse + CoLES; CB/LGB/multi-task ETT |
| `build_direct_next_generation_submission.py` | [RESULT](results/RESULT_SUBMISSION_NEXT_GEN_QUAD_V1.md), job bt1n2e9pa3592plu15bk, RESULT commit 498b5aa | Frequency specialist 0.40 + delta CB 0.28 + Two-Tower 0.17 + direct CB 0.15 |

Последний RESULT описывает Quad Stack и его log-space blend
`0.60 * champion_z + 0.40 * quad_z`. Champion input:
`submission_meta_blend_champions_v1.csv`. В отчёте для него указано **1.655904**;
это другое число, его нельзя округлением приравнять к **1.6558615631**.

В корне есть CSV нескольких meta-blend вариантов, но код сборки champion input,
его веса и связь с конкретной загрузкой не найдены в scripts/scratch, ноутбуках,
docs и просмотренной истории blend/champion commits. В нескольких manifests
job ID остался `__JOB_ID__`. Имена CSV не доказывают факт загрузки.

Pipeline последнего документированного обучения установлен; точный pipeline
итогового leaderboard submission **не установлен полностью**. Для восстановления
нужны receipt/идентификатор загрузки, SHA-256 отправленного CSV и manifest
финального meta-blend с input hashes/weights.

Ограничения воспроизведения:

- Raw train/template, banks, CSV и weights доступны локально, но вне Git.
- Исторические checkpoints не обязательно входят в clean clone.
- Не все dependencies pinned; neural seed не везде фиксируется.
- Текущий feature provider отличается от исходного 68f baseline; число фичей
  читать из конкретного manifest, не комментария builder.
- `fit_nonnegative_simplex_blend` в `blending.py` остаётся `NotImplementedError`;
  builders применяют уже зафиксированные числовые веса.
- Командный финальный score в ходе этой работы не воспроизводился.
