# RESULT — TCN screen: 180d, 2 epochs, LR 3e-4

## Identity

- Experiment: `post_ny_screen_tcn_h180_e2_lr3e4_v1`
- DataSphere job: `bt1bgpb3op7l7dncl0sn` (`SUCCESS`)
- PRE-RUN commit: `79e12df`
- Profile: `POST_NY_PUBLIC_PROXY`; RUN A meta anchor `2025-12-15`, RUN B validation anchor `2026-01-14`
- Cohort: deterministic 25k screen cohort, SHA-256 `eb9f9fcf949d59120cc8bddc972e8bcb9a77a0bea273ac120fdb3ffbdbe4cf3c`
- Train input SHA-256: `5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`
- Resolved config SHA-256: `2a4a3ec5a705ca1522cff36a220b03a5397b16e1a85be886267fc83e4968d761`

## Recipe and execution evidence

- TCN causal daily-tensor input: right-aligned full `180` days, 15 channels.
- Two full epochs: RUN A base `1078` steps / `550000` examples; RUN B base `1372` steps / `700000` examples.
- TCN React and Churn use their configured H/F specialist phases; every phase completed its requested step count.
- GPU: Tesla V100-PCIE-32GB, CUDA 12.1, PyTorch 2.3.1+cu121.
- Job-side pipeline elapsed time: `318.819` seconds (excludes DataSphere environment setup and output transfer).

## Artifact verification

- Individual-screen verifier: `OK`, 25,000 aligned rows.
- Candidate artifact aggregate SHA-256: `5dc3a0ebcc88502b889deba79a801f1a4e898fa1adb253132d5e42642cd73914`.
- RUN A incremental bank SHA-256: `e3f0687326eee52f06b1ae90984d503469926f69751f9dbb8fe3a7cb6a6b9082`.
- RUN B incremental bank SHA-256: `6f107bf8f0816c7b12e944a96214423361d0050deb5e48c31270d3bc1d7edfc2`.
- Frozen incremental meta SHA-256: `2c0d2562c2cc88e80c6182e03c0ed1199bff44d367bc462f65b7cb92e71e68ea`.

## Gate against immutable baseline v2

Baseline: RMSLE `1.6875360758674314`, MSE `2.847778007354049`.

Candidate: RMSLE `1.6875360666962527`, MSE `2.8477779764006597`.

- Delta MSE (candidate − baseline): `-0.0000000309533898`
- Approximate delta RMSLE: `-0.0000000091711787`
- Paired bootstrap (5,000 repeats): 95% MSE CI `[-0.0000000552642406, -0.0000000068958834]`; P(candidate better) `0.9934`.

The numerical gain is real under the paired screen sample but materially negligible. The frozen RUN A meta assigned `0.00577` to `tcn_react_logit` and `0` to `tcn_churn_logit`; TCN is **not promoted** from this point alone.

## Decision

Keep the candidate only as the central point of the TCN LR search. Next, run the same full-180d/two-epoch recipe at `1e-4` and `8e-4` sequentially, each through the same artifact and paired-gate checks. Do not alter history length before choosing an LR region.
