# RESULT — reference_framework_v1 SSL parity

Дата запуска: 2026-08-23 20:48:06–21:51:18 (локальное время)

## Job и provenance

- DataSphere job: `bt12tm634pu67l80k2rv`
- статус: `SUCCESS`
- PRE-RUN commit: `22f87864bab42adea9bb5965697b0f52acb1c131`
- profile: `POST_NY_PUBLIC_PROXY`
- experiment: `post_ny_ssl_parity_selected_100k_v1`
- cohort: immutable `selected_users_100k.parquet`, 100,000 users
- validation anchor: `2026-01-14`
- validation window: `2026-01-15 .. 2026-02-13`
- runtime: `1984.37 s` (33.07 min; VM/environment startup counted separately by DataSphere)

GPU подтверждён в run manifest: Tesla V100-PCIE-32GB, CUDA 12.1, torch `2.3.1+cu121`, 34,079,899,648 bytes.

## Parity result

| Run | RMSLE |
|---|---:|
| Frozen SSL baseline | 1.6811849063 |
| New config-driven framework | 1.6813296442 |
| Difference | +0.0001447379 (+0.0086%) |

RUN A meta-anchor RMSLE: `1.7409357099`.

The difference is within expected stochastic/implementation tolerance for an independent fresh training run. The new registry/config-driven framework reproduces the old four-model SSL validation behavior; no TCN, Residual MLP or BTYD model was enabled.

## Validation diagnostics

- rows: `100000`
- target buy rate: `0.53984`
- predicted buy rate: `0.576385`
- React AUC: `0.680157`
- Churn AUC: `0.804116`
- transition RMSLE: `00=0.626968`, `01=2.917733`, `10=2.249093`, `11=1.519984`

## Artifact hashes

Output root: `artifacts/reference_v1/experiments/post_ny_ssl_parity_selected_100k_v1/`

- config SHA256: `3baf98271ac823a20994ac18ab4173d808d8ec1ba6eaf7f16c4cb2e755d24576`
- train SHA256: `5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0`
- cohort SHA256: `d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2`
- RUN A bank: `1ce21b35d41a4557aab9e645d0c6e437daa770301c87dafcf9a321e903747a71`
- RUN B bank: `a855b75faa238bc2d55b34a8342f2b91d50a910a90b18d5432f2f80226f955f5`
- frozen meta: `f0040046bb1c1589b0aacff6ac24e1cfe6c740dd0671a21b588961197cba36cd`
- validation report: `207b5df7bf644b6a2f840148836ee525597e34bd0df833ad5b87c568bc7ccf43`

The DataSphere-generated manifest currently records `job_id: null` inside the downloaded JSON; this RESULT document is the authoritative link to job `bt12tm634pu67l80k2rv` and does not modify the generated artifact.
