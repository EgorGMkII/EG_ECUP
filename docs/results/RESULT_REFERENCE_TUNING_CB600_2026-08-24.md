# RESULT — CatBoost specialists, 600 trees

- Individual DataSphere job: `bt17k5grsj4besmmbupn`
- PRE-RUN: `2d97248ac7ef3805d7d319afb1750d3e77ae5c9b`
- 25k `POST_NY_PUBLIC_PROXY`, BTYD classifier features enabled, no direct branch.

| Evaluation | RMSLE |
| --- | ---: |
| Immutable six-model baseline | 1.6875360759 |
| Incremental meta gate with CatBoost-600 | 1.6876856855 |
| Gate delta vs baseline | +0.0001496097 |

600 trees improve over 300 (`1.6877594598`) but are still worse than the
immutable 1500-tree baseline.  The sequential bracket continues at 1000 trees.
