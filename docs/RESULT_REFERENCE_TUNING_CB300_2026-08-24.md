# RESULT — CatBoost specialists, 300 trees

- Individual DataSphere job: `bt11ss7b67nhotu3j1jf`
- Candidate PRE-RUN: `6b425d5df2390903f2303b2bd169fa7d895af25e`
- Incremental-gate PRE-RUN: `58de5552c87c992d6161ac050a421197d5a77a8b`
- Cohort/profile: 25k, `POST_NY_PUBLIC_PROXY`; BTYD classifier features enabled; direct branch disabled.
- Candidate job duration: 384.46 seconds (`6m24s`).

| Evaluation | RMSLE |
| --- | ---: |
| Standalone CatBoost hurdle, RUN B | 1.7011237597 |
| Immutable six-model baseline | 1.6875360759 |
| Incremental meta gate with CatBoost-300 | 1.6877594598 |
| Gate delta vs baseline | +0.00022338398 |

The 300-tree candidate is not promoted: its frozen-M meta gate is worse on V.
It remains a provenance-recorded lower bracket point; the next sequential
candidate is 600 trees.
