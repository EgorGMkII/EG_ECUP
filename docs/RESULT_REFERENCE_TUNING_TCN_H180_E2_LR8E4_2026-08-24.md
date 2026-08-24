# RESULT — TCN screen: 180d, 2 epochs, LR 8e-4

- Experiment: `post_ny_screen_tcn_h180_e2_lr8e4_v1`; DataSphere job `bt1ll7p5ov0q3i4oi069` (`SUCCESS`).
- PRE-RUN commit: `6debf8c`; config SHA-256: `60364fdd510c74ac8f9526efdccc14e256523242b53362a56ccb4c3d6ea9fb7a`.
- Fixed protocol: 25k deterministic screen cohort, causal full 180-day daily tensor, two epochs in each RUN, Tesla V100. All requested specialist and base budgets completed exactly.
- Individual artifact verification: `OK`, 25,000 aligned rows. Candidate aggregate SHA-256: `b9b1d5a988e7bffc0ac8ee122d03f63e3d9a9102cb6a15ef27a894c8c13e304e`.
- Job-side pipeline elapsed: `323.595` seconds.

## Gate against immutable baseline v2

| Metric | Baseline | Candidate | Candidate − baseline |
|---|---:|---:|---:|
| MSE | 2.847778007354049 | 2.847956711603621 | +0.000178704249572 |
| RMSLE | 1.6875360758674314 | 1.6875890233121396 | about +0.000052947444708 |

Paired bootstrap, 5,000 repeats: MSE 95% CI `[+0.000022393792557, +0.000336016204643]`; P(candidate better) `0.0122`.

RUN A meta gave `tcn_react_logit` weight `0.04321` and effectively zero Churn weight. Its V transfer is reliably harmful.

## LR-wave decision

The broad two-epoch/full-180d bracket is complete:

| Base LR | ΔMSE vs baseline | Gate reading |
|---|---:|---|
| `1e-4` | -0.0000035801 | CI crosses 0; TCN weights effectively 0 |
| `3e-4` | -0.0000000310 | numerically negligible; React weight 0.00577 |
| `8e-4` | +0.0001787042 | reliably harmful |

Reject `8e-4`. `1e-4` provides no TCN contribution. Retain `3e-4` only as the central LR for the next, tightly scoped epoch check (1 and 3 epochs); do not promote TCN from this LR wave.
