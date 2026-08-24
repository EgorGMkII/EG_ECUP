# RESULT — TCN screen: 120-day history, two epochs, LR 3e-4

## Scope

- Experiment: `post_ny_screen_tcn_h120_e2_lr3e4_v1`
- Stage/cohort: independent `screen`, deterministic 25k cohort
- Protocol: `POST_NY_PUBLIC_PROXY`; RUN A M=`2025-12-15`, RUN B V=`2026-01-14`
- Candidate: TCN React/Churn only; causal right-aligned 120-day view of the common daily store
- Job: `bt1l22uoc71gagtsqimm` (`SUCCESS`)
- PRE-RUN commit: `5e3959c`

## Exact execution

RUN A base completed `1078/1078` steps over 550,000 examples; RUN B base completed
`1372/1372` steps over 700,000 examples.  All React/Churn H/F specialist budgets completed.
Only the TCN's right-aligned history view differed from the 90/180-day checks.

## Gate against immutable baseline

| Metric | Baseline | Candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| MSE (log space) | 2.8477780074 | 2.8477739103 | -0.0000040971 |
| RMSLE | 1.6875360759 | 1.6875348619 | -0.0000012139 |

Paired bootstrap (5,000 repeats): MSE delta 95% CI `[-0.0000311128, +0.0000214171]`;
probability candidate better `0.6188`.  RUN A meta assigned React exactly `0` and Churn
`4.86e-18`, effectively zero. The small score movement is meta numerical re-fitting, not
a usable independent TCN signal.

## Provenance

- Individual artifact aggregate SHA-256: `e94a035360c9a1c4ed6719960c59a225fc4c9a28f04183a3af32bbcda502abfb`
- Individual prediction bank SHA-256: `94e1edfaafc1c206bb142dbb88203696f723e5eab408fde4b32a8bb0c849d8b1`
- Frozen incremental meta SHA-256: `f8a21700589ed2cbb28d546e706ab9014ab4521ed9a5b3122bc237b2470c1457`
- Resolved config SHA-256: `fa284a743e14f4ec9c3dbdc462850544efacff0428b9c542fd36b507995e54c9`

## TCN search conclusion

The bracket is complete: LR `1e-4`, `3e-4`, `8e-4`; 1/2/3 epochs at 180d; and 90/120/180d
at two epochs and `3e-4`. None earned a material positive paired gate.  180d/3epochs had a
non-zero React contribution but was harmful; the two-epoch history variants were zero-weight.
Do not add TCN to the selected final stack. Proceed to standalone and late-blend direct
CatBoost evaluation.
