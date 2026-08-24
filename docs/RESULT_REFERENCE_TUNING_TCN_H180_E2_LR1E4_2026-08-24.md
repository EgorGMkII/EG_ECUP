# RESULT — TCN screen: 180d, 2 epochs, LR 1e-4

- Experiment: `post_ny_screen_tcn_h180_e2_lr1e4_v1`; DataSphere job `bt182jg7de1sh85a5gmk` (`SUCCESS`).
- PRE-RUN commit: `4711d44`; config SHA-256: `f55c43e7159f9d507c2b9e2548b2604a926647f205e6ff8eb746c5352c1de1eb`.
- Same deterministic 25k cohort/profile/180-day causal input and two-epoch budget as the LR `3e-4` central candidate. Only TCN base LR differs.
- Pipeline runtime: job completed successfully with exact RUN A/RUN B budgets; individual artifact verification is `OK` for 25,000 aligned rows. Candidate aggregate SHA-256: `f3e432de7fbba252efb35950cdcb719bd3daa99c6549c7e8666f09dce6dfca86`.

## Gate against immutable baseline v2

| Metric | Baseline | Candidate | Candidate − baseline |
|---|---:|---:|---:|
| MSE | 2.847778007354049 | 2.847774427280446 | -0.000003580073603 |
| RMSLE | 1.6875360758674314 | 1.6875350151272257 | about -0.000001060740206 |

Paired bootstrap, 5,000 repeats: MSE 95% CI `[-0.000030619489716, 0.000021856346026]`; P(candidate better) `0.6042`.

RUN A meta assigned effectively zero weights to both TCN outputs (`tcn_react_logit=6.31e-18`, `tcn_churn_logit=1.01e-17`). Thus the apparent V difference is only numerical movement in the re-fit baseline meta, not evidence that this TCN recipe contributes to the stack.

## Decision

`LR=1e-4` is not promoted. It remains one endpoint of the planned broad LR bracket. Run `LR=8e-4` next with the identical full-180d/two-epoch protocol; do not select an epoch budget or history length until that bracket is closed.
