# RESULT — 25k no-direct epoch baseline v2

- DataSphere job: `bt1ov9kj67aslah740ub`
- PRE-RUN commit: `f6398cf2b144c20ce80ad9f59928d4c01d8842fa`
- Profile: `POST_NY_PUBLIC_PROXY`; screen cohort: 25,000 users.
- Stack: CatBoost specialists + S1 + S2 + ETT + TCN + Residual MLP.
- Direct CatBoost: disabled by design for the individual-model tuning stage.
- Device: Tesla V100-PCIE-32GB.
- DataSphere elapsed: 2,349.165 seconds (`39m09s`); measured pipeline elapsed: 1,678.375 seconds.

## Immutable baseline

| Metric | Value |
| --- | ---: |
| RUN A M RMSLE | 1.7567415486 |
| RUN B V RMSLE | 1.6875360759 |
| RUN B log-space MSE | 2.8477780074 |
| 00 RMSLE | 0.6784047000 |
| 01 RMSLE | 2.8660910533 |
| 10 RMSLE | 2.2434195201 |
| 11 RMSLE | 1.5326892420 |

## Provenance

- Resolved config SHA-256: `8d3d6a74f4dfe0a07a98066152f1ad0323c28bdd6e5d9e4f6139a9317d445890`
- RUN B prediction-bank SHA-256: `bbb307682fd9d208ebb984e512288cf377bb0bfca84d4169f05ddf8cc91d6dd4`
- Frozen meta SHA-256: `5a27b697a70e00418de6c2fdbcc39c2cf542ef1e7a35287e872ac27fae70e07a`
- Artifact-manifest SHA-256: `cf4367b8fc42d1999193bd6f48e5b942998804b112cc585f1b8a2d44bf20efda`

`verify_reference_experiment_artifacts.py` returned `OK`: all eight output hashes,
both 25k banks, feature order, user order, frozen meta provenance and validation
provenance match.  This is the immutable baseline for the CatBoost 300/600/1000/1500
individual screens and their incremental meta gates.
