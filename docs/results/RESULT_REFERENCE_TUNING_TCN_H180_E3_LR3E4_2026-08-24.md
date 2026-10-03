# RESULT — TCN screen: 180-day history, three epochs, LR 3e-4

## Scope

- Experiment: `post_ny_screen_tcn_h180_e3_lr3e4_v1`
- Stage/cohort: independent `screen`, deterministic 25k cohort
- Protocol: `POST_NY_PUBLIC_PROXY`; RUN A M=`2025-12-15`, RUN B V=`2026-01-14`
- Candidate: TCN React/Churn only; 180-day right-aligned daily history
- Job: `bt16hjag4sfesb6cv5u6` (`SUCCESS`)
- PRE-RUN commit: `e810907`

## Exact execution

TCN base executed three complete epochs: RUN A `1617/1617` steps over 825,000 examples;
RUN B `2058/2058` steps over 1,050,000 examples.  React and Churn H/F specialist passes also
completed their exact one-epoch budgets.  RUN A and RUN B created independent model,
optimizer and scheduler objects.

## Gate against immutable baseline

| Metric | Baseline | Candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| MSE (log space) | 2.8477780074 | 2.8478097548 | +0.0000317474 |
| RMSLE | 1.6875360759 | 1.6875454823 | +0.0000094064 |

Paired bootstrap (5,000 repeats): MSE delta 95% CI `[-0.0000371016, +0.0001031349]`;
probability candidate better `0.1902`. The frozen RUN A meta gave TCN React a `0.0218153`
weight and Churn `0`, so this is a real but harmful React contribution rather than a
zero-weight numerical re-fit.

## Provenance

- Individual artifact aggregate SHA-256: `51a3f864885f2a1eddac800ffbf92b53c42c0bc694d16eb5706700c59f5582fe`
- Individual prediction bank SHA-256: `5e27510396f8b55b74aab7243fd27e53355b9a5ba72496df63543bc0e6295f05`
- Frozen incremental meta SHA-256: `dc40fabd7948dfbfa382106c11a9d22d789370c1057acde28aae1e6276d312cc`
- Resolved config SHA-256: `e7bd1a7bcc9efbca53e7323da2918b62d08b4b99ac9f6258c51cf3871c125b83`

## Decision

Reject 180-day/three-epoch TCN.  The 180-day epoch bracket is complete: one and two epochs
were effectively zero-weight/no material effect, while three epochs produced a small harmful
React contribution.  Proceed with the predeclared 90- and 120-day history checks at two
epochs and LR `3e-4`; do not promote any TCN configuration without a paired, material gain.
