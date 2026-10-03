# RESULT — TCN screen: 180-day history, one epoch, LR 3e-4

## Scope

- Experiment: `post_ny_screen_tcn_h180_e1_lr3e4_v1`
- Stage/cohort: independent `screen`, deterministic 25k cohort
- Protocol: `POST_NY_PUBLIC_PROXY`; RUN A M=`2025-12-15`, RUN B V=`2026-01-14`
- Candidate: TCN React/Churn only; 180-day right-aligned daily history
- Job: `bt1r01kdfqna5kf4lrsd` (`SUCCESS`)
- PRE-RUN commit: `6c8febc`

## Execution

The base pass used exactly one epoch: RUN A `539/539` and RUN B `686/686` optimizer steps.
Specialists retained their fixed one-epoch H/F schedules.  The job produced the normal 25k
prediction bank and diagnostics; no historical artifact was overwritten.

## Gate against immutable baseline

| Metric | Baseline | Candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| MSE (log space) | 2.8477780074 | 2.8477743857 | -0.0000036217 |
| RMSLE | 1.6875360759 | 1.6875350028 | -0.0000010731 |

Paired bootstrap (5,000 repeats): MSE delta 95% CI `[-0.0000306810, +0.0000218210]`;
probability candidate better `0.6048`. The TCN meta weights were effectively zero
(React `7.60e-18`, Churn `3.51e-18`), so the tiny score movement is numerical re-fitting
of the baseline meta solution, not a usable TCN effect.

## Provenance

- Individual artifact aggregate SHA-256: `0130b74f23dd4bd796df4628bd57a376702a88c65ec1879afea943c56b7e5895`
- Individual prediction bank SHA-256: `82770a30b1e365467b75534edb0b9b2998e39f9ab3f74218ff2d892780408bf9`
- Frozen incremental meta SHA-256: `0bd0fb846f9337102397eae52059dc85cad962c2dc9f84be1c99ad4e53aeb58d`
- Resolved config SHA-256: `033e66767c7df7713548c3594d1ed8ed3414ab1fa6ad1b45812932c5444bb611`

## Decision

Do not promote this candidate. Run the predeclared three-epoch endpoint at the same 180-day
history and LR `3e-4`; then select the history-length follow-up only if a configuration earns
a material, statistically supported TCN meta contribution.
