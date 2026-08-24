# RESULT — TCN screen: 90-day history, two epochs, LR 3e-4

## Scope

- Experiment: `post_ny_screen_tcn_h90_e2_lr3e4_v1`
- Stage/cohort: independent `screen`, deterministic 25k cohort
- Protocol: `POST_NY_PUBLIC_PROXY`; RUN A M=`2025-12-15`, RUN B V=`2026-01-14`
- Candidate: TCN React/Churn only; causal right-aligned 90-day view of the common daily store
- Job: `bt146121ckq31mui51n5` (`SUCCESS`)
- PRE-RUN commit: `2579644`

## Execution

Base budgets completed exactly: RUN A `1078/1078` steps (550,000 examples), RUN B
`1372/1372` steps (700,000 examples).  React/Churn H/F budgets also completed exactly.
The store source, anchors, cohort, seed and all non-TCN recipes matched the immutable baseline.

## Gate against immutable baseline

| Metric | Baseline | Candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| MSE (log space) | 2.8477780074 | 2.8477744385 | -0.0000035688 |
| RMSLE | 1.6875360759 | 1.6875350185 | -0.0000010574 |

Paired bootstrap (5,000 repeats): MSE delta 95% CI `[-0.0000306363, +0.0000219148]`;
probability candidate better `0.6038`.  Frozen RUN A meta assigned exactly zero to both TCN
React and TCN Churn.  The tiny movement is therefore baseline-meta numerical re-fitting, not
a usable TCN signal.

## Provenance

- Individual artifact aggregate SHA-256: `1ce7d8b2223680979832d2438185f260ec6b0ef10f8840a49d1a5c396e0a4b82`
- Individual prediction bank SHA-256: `ec52027ea0e17b3ffd8eac3fd240f49b93857261f2a5e434c692e3cbfff9800f`
- Frozen incremental meta SHA-256: `6c22c140c268c6a83feb13784106d62c679fa40b9582ec6b0548f2377309f217`
- Resolved config SHA-256: `c9850328743eb4837b1f8f5ec4eaa40e0164b332d118e063d9a49cd17f8fb978`

## Decision

Do not promote 90-day TCN. Run the predeclared 120-day / two-epoch / LR `3e-4` candidate as
the final history-length check, then close the TCN search without adding any zero- or
negative-contribution configuration to the final stack.
