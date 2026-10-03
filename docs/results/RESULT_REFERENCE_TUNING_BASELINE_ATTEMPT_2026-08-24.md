# Failed 25k epoch baseline attempt — 2026-08-24

- DataSphere job: `bt15a8n1044rau9fvvv1`
- PRE-RUN commit: `4c75d0f17fbe3ea5f88912496f1eb94d10e1efbc`
- Profile/cohort: `POST_NY_PUBLIC_PROXY`, nested screen cohort (25,000 users)
- Stack: CatBoost, S1, S2, ETT, TCN, Residual MLP; `catboost_direct` excluded.
- GPU: Tesla V100-PCIE-32GB; CUDA was available.
- DataSphere timestamps: created `2026-08-24T06:35:22.235Z`, error `2026-08-24T06:49:16.495Z`.
- Python elapsed time before failure: `168.640770805` seconds.

## Failure classification

The downloaded `run_manifest.json` records `NameError: name 'recipe' is not defined`.
The error came from the newly parameterized GRU/ETT specialist phase helpers:
they read `recipe.synchronized_epochs` without receiving `recipe` as an
argument.  No validation metric, prediction bank, meta package, or candidate
comparison was produced.  This attempt is therefore excluded from the tuning
leaderboard.

## Corrective action

The follow-up PRE-RUN must include the fix that explicitly propagates the
optional recipe to both specialist-phase helpers and covers the behavior with
CPU regression tests.  The retry uses the same immutable experiment config and
creates no overwrite of this failed attempt's artifacts.
