# RESULT: SSL_TEMPORAL_STACK_V1 post-NY validation

- DataSphere GPU smoke job: `bt1nnrcov4knb6rrfc83`
- DataSphere full job: `bt1rsfqtn9jj4e52blom`
- PRE-RUN commit: `3c25ab32af42a5ab85198207aa6dccf3af1fd09e`
- Status: `SUCCESS`
- Full DataSphere wall time: 61 minutes 41 seconds
- Pipeline elapsed time: 3411.408 seconds (56 minutes 51 seconds)
- GPU: Tesla V100-PCIE-32GB, CUDA 12.1, PyTorch 2.3.1+cu121
- Experiment: clean `SSL_TEMPORAL_STACK_V1`; this is not the historical Public builder

## Protocol

RUN A used 11 training anchors from `2025-06-23` through `2025-11-10` and
predicted meta anchor `M=2025-12-15`. RUN B created all first-level models from
scratch, used 14 training anchors from `2025-06-23` through `2025-12-15`, and
predicted validation anchor `V=2026-01-14`. The validation target window is
`2026-01-15 .. 2026-02-13`.

The stack order was CatBoost, S1, S2, and ETT. S1 and S2 each used 750 SSL
steps and 2000 joint-base steps in both runs. ETT used 3500 joint-base steps.
Every neural model then trained independent React, Churn, and Amount
specialists with 400 frozen-head steps and 600 fine-tuning steps. RUN A models
were discarded before RUN B; only the frozen RUN A meta package was applied to
RUN B predictions.

## Metrics

| Metric | Value |
| --- | ---: |
| RUN A meta RMSLE | `1.7409277438` |
| RUN B validation RMSLE | `1.6811849063` |
| RUN B validation MSE in log space | `2.8263826893` |
| Target buy rate | `0.539840` |
| Predicted buy rate | `0.576374` |
| Target mean log value | `2.236402` |
| Prediction mean log value | `2.380509` |

Transition diagnostics:

| Transition | Rows | RMSLE | Share of total squared error |
| --- | ---: | ---: | ---: |
| 00 | 21,150 | `0.6270893422` | `2.94%` |
| 01 | 3,848 | `2.9175201453` | `11.59%` |
| 10 | 24,866 | `2.2468697820` | `44.42%` |
| 11 | 50,136 | `1.5213052106` | `41.05%` |

Specialist diagnostics:

- React: AUC `0.680126`, logloss `0.407048`, Brier `0.124080` on 24,998 rows.
- Churn: AUC `0.804083`, logloss `0.500786`, Brier `0.168142` on 75,002 rows.
- Amount: conditional-z RMSE `1.124823` on 53,984 buyer rows.

This score is the canonical internal baseline for subsequent experiments that
use exactly this temporal protocol. It must not be compared as if it were a
Public LB score: the historical fixed-meta Public builder remains the Public
baseline at RMSLE `1.663815381013448`.

## Acceptance checks

- GPU smoke completed in 38.135 seconds on `cuda:0` and verified CatBoost GPU,
  CUDA neural training, FP16 event memmap, streaming logs, and isolated output
  download.
- Full job returned all seven declared result files.
- Both prediction banks contain 100,000 unique cohort users in exact cohort
  order, 18 expected columns, and only finite numeric values.
- Reconstructed RUN B meta components are finite. Final `prediction_z` is
  non-negative and ranges from `0.1217339765` to `7.5478862721`.
- All 46 reported neural phases have exact requested/completed step parity.
- All six CatBoost task reports resolve to `task_type="GPU"`, device `0`, and
  1500 trees.
- The immutable cohort hash, raw-train hash, config hash, commit SHA, and every
  declared artifact hash were verified locally after download.

The DataSphere process does not expose its job ID inside the guest environment,
so `job_id` is `null` in the immutable in-job JSON files. This RESULT record
binds those files to full job `bt1rsfqtn9jj4e52blom`; the artifacts themselves
were not rewritten after download.

## Input and configuration hashes

| Input | SHA-256 |
| --- | --- |
| `data/train.parquet` | `5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0` |
| `selected_users_100k.parquet` | `d618e98744302eeec7352b6dc2f2db4f1b127298f1b84b6918304cf3368c4fd2` |
| experiment config | `a5827d17403ea659d58c89ee5c5fedf56ec27c52812eb5950c81fb10ec9de61f` |
| CatBoost feature order | `e54a385cd69665f09063762a2e8581fe953b759004a69180487dfcea3fa8df6e` |

## Full-job artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `frozen_meta_package.json` | `9599763b71ef0003ae766c47617a6ac3a277f34b4b9901835324ac86d6405076` |
| `model_training_report.json` | `03439959cae11a8a8482143231498e9caece18edb66e845ff0a46f8e1c2504fd` |
| `run_a_meta_prediction_bank.parquet` | `b1b944cb7b5cb6bb02deadd2fe2f3eb62add5c735d91bd53481ad0bbf46e2825` |
| `run_b_validation_prediction_bank.parquet` | `ca3e0bf4f41fa34ff492292b8b4b2204a0c63085c07b7e6c853a609f338c13e1` |
| `run_manifest.json` | `54a7506bdb7fcfb46d8ceac6a3223876adae8b91dcf597b9a82e2bddd0b7e928` |
| `validation_report.json` | `97626366b3a862f4f65bcd8bb137340e92adc585e45935c621e5ccb430481c0d` |

## GPU smoke artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `smoke_manifest.json` | `66c244f8dc596800e25fdaa5503ffa0b48127b939b12853c4c5c43e453411473` |
| `smoke_training_report.json` | `1fba6ab52b35ec79f4bbc3c19b569e0c1ca7d6ef18bd32b838c31bc5b97b971f` |
| `smoke_prediction_bank.parquet` | `8b738709cfe4476329bcec53f31d341203e09578e9d94db6b3a882f4175b427a` |
| `smoke_meta_package.json` | `30b4baa094c162104ea0e2a3b118e98baf2fbe297f1dcea08d5328a45b8e2742` |
| `smoke_report.json` | `78bf46fb255b61c334d2015668c0a2b10b38e8fe315e24bb265900fe72b85a50` |
