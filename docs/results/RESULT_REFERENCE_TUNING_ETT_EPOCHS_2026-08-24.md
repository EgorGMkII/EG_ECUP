# ETT epoch screen — 2026-08-24

Immutable 25k screen baseline: `post_ny_epoch_tuning_screen_baseline_v2`.

| Candidate | DataSphere job | RUN B stack RMSLE | Δ vs baseline | Decision |
| --- | --- | ---: | ---: | --- |
| Baseline ETT recipe | `bt1ov9kj67aslah740ub` | 1.6875360759 | 0 | incumbent |
| 2 epochs, LR 3e-4 | `bt1grvieb8eeip8eev4b` | 1.6876173810 | +0.0000813051 | reject |
| 3 epochs, LR 3e-4 | `bt1hv1pvfrf95e8nc48h` | 1.6873598923 | -0.0001761836 | inconclusive |
| 2 epochs, LR 2e-4 | `bt13kc99tveur2kpu2qo` | 1.6884213494 | +0.0008852735 | reject |

The 3-epoch candidate was prepared at commit `bc1750bb9e6b0b4d1d6a9849199a5a1112841c58` with resolved-config SHA256 `a1d28c2ee141cae7f831ebb043add43d02c74c99551154d7fad5ad6437c9a75a`.
It used independent RUN A and RUN B, respectively 1617 and 2058 base optimizer steps after epoch expansion. Its candidate-bank and gate artifacts passed SHA/schema/user-order verification.

Paired bootstrap (5,000 draws) for the 3-epoch candidate against the immutable baseline: ΔMSE `-0.0005946012`, 95% CI `[-0.0016415530, 0.0004559203]`, probability of improvement `0.8670`. The interval crosses zero, so the candidate is not promoted.

The only adaptive follow-up, 2 epochs at LR `2e-4`, was materially worse. The ETT epoch/LR wave is therefore closed with the baseline ETT recipe retained; no further ETT configs will be launched in this wave.
